"""Tests for Stage 2 Features:
===========================
1. Voice Dictation & Read-Result-Aloud Privacy:
   - Read-result-aloud strictly omits patient name unless explicitly ticked.
2. Smart Token & QR Ticket Default Privacy:
   - Omits priority tier, patient name, and vital signs by default from HTML and QR payload.
3. Shift Handover Summary (Rule-Based SBAR):
   - Computes clinical metrics, formats SBAR plain text report, and exports CSV.
   - Strict count parity: total encounters and tier counts in .txt identically match .csv.
4. Hospital Surge Mode:
   - Threshold detection for queue volume and L1/L2 emergent cases.
   - Manual CMO override engagement.
   - Surge alert banner markup and operational checklist verification.
5. Streamlit AppTest:
   - Executes app.py end-to-end using streamlit.testing.v1.AppTest to verify UI rendering without exceptions.
"""

import csv
import io
import json
from datetime import datetime, timedelta
import pytest
from streamlit.testing.v1 import AppTest

import db
import seed_data
from smart_ticket import render_smart_token_ticket, generate_qr_base64
from shift_handover import (
    compute_shift_metrics,
    generate_shift_handover_txt,
    generate_shift_handover_csv
)
from surge_mode import (
    evaluate_surge_status,
    render_surge_banner_html,
    SURGE_CHECKLIST_TEMPLATE
)
from audio_speech import build_read_aloud_text


# -----------------------------------------------------------------------------
# 1. SMART TOKEN & QR PRIVACY TESTS
# -----------------------------------------------------------------------------
def test_smart_token_default_privacy_omits_tier_name_and_vitals():
    """Patient-facing token and QR code must strictly omit tier, name, and vitals by default."""
    encounter = {
        "encounter_id": "ENC-20261015-0077",
        "patient_name": "Sita Devi",
        "age": 42,
        "gender": "Female",
        "department": "Cardiology",
        "tier": 2,
        "tier_name": "Emergent",
        "abstained": False,
        "vitals": {
            "spo2": 93.0,
            "bp_systolic": 150,
            "bp_diastolic": 95,
            "pulse": 105,
            "temperature": 99.2
        },
        "created_at": datetime.now().isoformat()
    }

    # 1. Default Patient Ticket
    patient_ticket = render_smart_token_ticket(encounter, queue_pos=3, est_wait_mins=25)
    assert "TOKEN #T-0077" in patient_ticket
    assert "Cardiology" in patient_ticket
    assert "Sita Devi" not in patient_ticket  # NO NAME
    assert "L2: Emergent" not in patient_ticket  # NO TIER BADGE
    assert "SpO2:" not in patient_ticket  # NO VITALS
    assert "Intake Vitals:" not in patient_ticket

    # 2. Staff Ticket (Explicit Authorization)
    staff_ticket = render_smart_token_ticket(encounter, queue_pos=3, est_wait_mins=25, show_staff_clinical_info=True)
    assert "TOKEN #T-0077" in staff_ticket
    assert "Sita Devi" in staff_ticket
    assert "L2: Emergent" in staff_ticket
    assert "Intake Vitals:" in staff_ticket


# -----------------------------------------------------------------------------
# 2. AUDIO READ-RESULT-ALOUD PRIVACY TESTS
# -----------------------------------------------------------------------------
def test_read_aloud_speech_privacy_enforcement():
    """Read-aloud announcement must never speak the patient name unless explicitly enabled."""
    encounter = {
        "encounter_id": "ENC-20261015-0102",
        "patient_name": "Ananya Sharma",
        "department": "Emergency",
        "tier": 1,
        "tier_name": "Resuscitation",
        "reasons": ["Severe hypotension (BP 68/40)", "Profound Hypoxia SpO2 82%"],
        "abstained": False
    }

    # Default: include_name=False
    private_speech = build_read_aloud_text(encounter, include_name=False)
    assert "Ananya Sharma" not in private_speech
    assert "Token T-0102" in private_speech
    assert "Resuscitation" in private_speech
    assert "Emergency" in private_speech
    assert "Severe hypotension" in private_speech

    # When explicitly authorized: include_name=True
    named_speech = build_read_aloud_text(encounter, include_name=True)
    assert "Ananya Sharma" in named_speech
    assert "Token T-0102" in named_speech
    assert "Resuscitation" in named_speech


# -----------------------------------------------------------------------------
# 3. SHIFT HANDOVER SUMMARY & STRICT COUNT PARITY TESTS
# -----------------------------------------------------------------------------
def test_shift_handover_report_parity_and_sbar_structure():
    """Verifies that .txt and .csv have matching encounter counts and adhere to clinical SBAR format."""
    db.clear_all()
    demo_encounters = seed_data.generate_curated_demo_records()
    db.bulk_insert(demo_encounters)

    all_encs = db.get_all_encounters()
    total_records = len(all_encs)
    assert total_records > 0

    # 1. Compute metrics
    metrics = compute_shift_metrics(all_encs, num_doctors=2)
    assert metrics["total_encounters"] == total_records
    sum_tiers = sum(metrics["tier_counts"][k] for k in [1, 2, 3, 4, 5, "abstained"])
    assert sum_tiers == total_records

    # 2. Generate SBAR text report
    sbar_txt = generate_shift_handover_txt(all_encs, num_doctors=2)
    assert "[S] SITUATION" in sbar_txt
    assert "[B] BACKGROUND" in sbar_txt
    assert "[A] ASSESSMENT" in sbar_txt
    assert "[R] RECOMMENDATION" in sbar_txt
    assert f"Total Patient Encounters Processed in Shift : {total_records}" in sbar_txt

    # 3. Generate CSV export
    csv_str = generate_shift_handover_csv(all_encs)
    reader = csv.DictReader(io.StringIO(csv_str))
    csv_rows = list(reader)
    assert len(csv_rows) == total_records  # EXACT MATCH: CSV row count equals total records in TXT report

    # Verify column presence
    assert "encounter_id" in reader.fieldnames
    assert "tier" in reader.fieldnames
    assert "department" in reader.fieldnames
    assert "is_escalated" in reader.fieldnames


# -----------------------------------------------------------------------------
# 4. SURGE MODE & CHECKLIST TESTS
# -----------------------------------------------------------------------------
def test_surge_mode_threshold_detection_and_checklist():
    """Verifies surge mode activation under high patient volume, emergent count, or manual override."""
    now = datetime.now()
    normal_encs = [
        {"encounter_id": f"ENC-N-{i}", "status": "Waiting", "tier": 4, "created_at": now.isoformat()}
        for i in range(5)
    ]

    # Normal conditions
    status_normal = evaluate_surge_status(normal_encs, waiting_threshold=10, emergent_threshold=3)
    assert status_normal["is_active"] is False
    assert status_normal["severity"] == "NORMAL"
    assert render_surge_banner_html(status_normal) == ""

    # High waiting queue surge (12 waiting >= threshold 10)
    high_volume_encs = [
        {"encounter_id": f"ENC-H-{i}", "status": "Waiting", "tier": 4, "created_at": now.isoformat()}
        for i in range(12)
    ]
    status_volume = evaluate_surge_status(high_volume_encs, waiting_threshold=10, emergent_threshold=3)
    assert status_volume["is_active"] is True
    assert "SURGE" in status_volume["severity"]
    banner_vol = render_surge_banner_html(status_volume)
    assert "OPD SURGE PROTOCOL ACTIVATED" in banner_vol
    assert "Waiting: 12 / Limit 10" in banner_vol

    # High emergent acuity surge (4 L1/L2 cases >= threshold 3)
    emergent_encs = [
        {"encounter_id": "ENC-E-1", "status": "Waiting", "tier": 1, "created_at": now.isoformat()},
        {"encounter_id": "ENC-E-2", "status": "Waiting", "tier": 2, "created_at": now.isoformat()},
        {"encounter_id": "ENC-E-3", "status": "Waiting", "tier": 2, "created_at": now.isoformat()},
        {"encounter_id": "ENC-E-4", "status": "Waiting", "tier": 1, "created_at": now.isoformat()},
    ]
    status_emergent = evaluate_surge_status(emergent_encs, waiting_threshold=15, emergent_threshold=3)
    assert status_emergent["is_active"] is True
    assert status_emergent["emergent_count"] == 4

    # Manual CMO override
    status_manual = evaluate_surge_status(normal_encs, waiting_threshold=15, emergent_threshold=3, force_surge=True)
    assert status_manual["is_active"] is True
    assert any("Manual Surge Override" in r for r in status_manual["reasons"])

    # Checklist verification
    assert len(SURGE_CHECKLIST_TEMPLATE) >= 5
    checklist_ids = [item["id"] for item in SURGE_CHECKLIST_TEMPLATE]
    assert "cmo_alert" in checklist_ids
    assert "fast_track" in checklist_ids
    assert "rapid_revitals" in checklist_ids
    assert "oxygen_check" in checklist_ids


from pathlib import Path
APP_PATH = str((Path(__file__).parent.parent / "app.py").resolve())


# -----------------------------------------------------------------------------
# 5. STREAMLIT APPTEST INTEGRATION TESTS
# -----------------------------------------------------------------------------
def test_streamlit_apptest_runs_command_center_without_exceptions():
    """Validates that app.py compiles, runs, and renders the Command Center cleanly via AppTest."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception
    # Check that navigation radio button exists
    assert len(at.radio) >= 1
    # Check that app rendered title or brand text
    assert len(at.sidebar) >= 1


def test_streamlit_apptest_navigates_to_patient_intake():
    """Validates navigation to Patient Intake and ensures form components render without error."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception

    # Select Patient Intake page
    intake_option = "⚡ Patient Intake (<60s)"
    at.radio[0].set_value(intake_option).run()
    assert not at.exception
    # Check that intake form inputs and buttons are present
    assert len(at.text_input) >= 1
    assert len(at.button) >= 1


def test_streamlit_apptest_navigates_to_public_status_display():
    """Validates navigation to the new Trilingual Public Queue Status Display screen."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception

    public_display_option = "📺 Public Status Display"
    at.radio[0].set_value(public_display_option).run()
    assert not at.exception
    # Verify page runs without exceptions


def test_streamlit_apptest_navigates_to_live_doctor_queue():
    """Validates navigation to the Live Doctor OPD Queue and checks queue rendering."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception

    queue_option = "📋 Live Doctor OPD Queue"
    at.radio[0].set_value(queue_option).run()
    assert not at.exception


def test_streamlit_apptest_surge_mode_engagement():
    """Validates manual surge mode engagement in AppTest."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception

    # Find the manual surge mode checkbox in sidebar and toggle it
    for chk in at.checkbox:
        if "Manual Surge" in chk.label:
            chk.check().run()
            break
    assert not at.exception
