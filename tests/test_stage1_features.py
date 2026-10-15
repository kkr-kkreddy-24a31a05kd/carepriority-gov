"""Tests for Stage 1 Features:
===========================
1. Vitals Fix: all vitals start empty as None; untouched form triggers Safety Abstention.
2. Smart Token + QR Ticket: base64 data-URI generation, ticket layout, and security payload.
3. Re-Triage Alerts: threshold detection per clinical tier, metrics tracking, and deterioration escalation.
"""

import pytest
from datetime import datetime, timedelta
import db
from triage_engine import run_triage, validate_vitals
from smart_ticket import generate_qr_base64, render_smart_token_ticket


def test_vitals_untouched_form_starts_none_and_abstains():
    """An untouched form with all vitals starting as None must trigger safety abstention."""
    # Validate vitals permits None
    is_valid, errs, warns = validate_vitals(
        age=None,
        spo2=None,
        bp_systolic=None,
        bp_diastolic=None,
        pulse=None,
        temperature=None
    )
    assert is_valid is True
    assert len(errs) == 0

    # Triage run with all None vitals and empty complaint must abstain
    res = run_triage(
        patient_id="TEST-UNTOUCHED-01",
        patient_name="",
        age=None,
        gender="Unknown",
        spo2=None,
        bp_systolic=None,
        bp_diastolic=None,
        pulse=None,
        temperature=None,
        complaint=""
    )
    assert res["abstained"] is True
    assert res["tier"] is None
    assert res["tier"] != 5
    assert "SAFETY ABSTENTION" in res["abstention_reason"]


def test_smart_token_qr_generation():
    """Smart Token Ticket generates valid base64 QR code and full ticket markup."""
    sample_encounter = {
        "encounter_id": "ENC-20261014-0042",
        "patient_name": "Ramesh Kumar",
        "age": 48,
        "gender": "Male",
        "department": "Cardiology",
        "tier": 2,
        "abstained": False,
        "vitals": {"spo2": 95.0, "bp_systolic": 140, "bp_diastolic": 90, "pulse": 88, "temperature": 98.4},
        "created_at": datetime.now().isoformat()
    }

    # Generate QR data URI
    qr_uri = generate_qr_base64({"token": "T-0042", "patient": "Ramesh Kumar"})
    assert qr_uri.startswith("data:image/")
    assert len(qr_uri) > 100

    # Render Staff Ticket (with explicit authorization)
    staff_ticket_html = render_smart_token_ticket(sample_encounter, queue_pos=2, est_wait_mins=24, show_staff_clinical_info=True)
    assert "TOKEN #T-0042" in staff_ticket_html
    assert "Ramesh Kumar" in staff_ticket_html
    assert "Cardiology" in staff_ticket_html
    assert "EMERGENCY WARNING" in staff_ticket_html
    assert "आपातकालीन सूचना" in staff_ticket_html
    assert qr_uri[:30] in staff_ticket_html or "data:image/" in staff_ticket_html

    # Render Default Patient Ticket (Privacy Safe: no name, no tier, no vitals)
    patient_ticket_html = render_smart_token_ticket(sample_encounter, queue_pos=2, est_wait_mins=24)
    assert "TOKEN #T-0042" in patient_ticket_html
    assert "Cardiology" in patient_ticket_html
    assert "Ramesh Kumar" not in patient_ticket_html
    assert "SpO2:" not in patient_ticket_html
    assert "L2: Emergent" not in patient_ticket_html


def test_retriage_threshold_detection_and_escalation():
    """Validates Re-Triage alert threshold detection and patient deterioration escalation."""
    db.clear_all()
    now = datetime.now()

    # Create L2 patient who has been waiting 20 minutes (threshold is 15m -> DUE)
    l2_enc = {
        "encounter_id": "ENC-RT-L2",
        "patient_name": "Waiting L2",
        "tier": 2,
        "tier_name": "Emergent",
        "department": "Emergency",
        "status": "Waiting",
        "abstained": False,
        "created_at": (now - timedelta(minutes=20)).isoformat(),
        "vitals": {"spo2": 94.0, "bp_systolic": 110, "bp_diastolic": 70, "pulse": 90, "temperature": 98.6},
        "audit": []
    }
    db.insert_encounter(l2_enc)

    # Create L3 patient who has been waiting 20 minutes (threshold is 60m -> NOT DUE YET)
    l3_enc = {
        "encounter_id": "ENC-RT-L3",
        "patient_name": "Waiting L3",
        "tier": 3,
        "tier_name": "Urgent",
        "department": "General Medicine",
        "status": "Waiting",
        "abstained": False,
        "created_at": (now - timedelta(minutes=20)).isoformat(),
        "vitals": {"spo2": 97.0, "bp_systolic": 120, "bp_diastolic": 80, "pulse": 75, "temperature": 98.6},
        "audit": []
    }
    db.insert_encounter(l3_enc)

    # Check threshold detection
    is_due_l2, wait_l2, thresh_l2 = db.check_retriage_due(l2_enc, now=now)
    assert is_due_l2 is True
    assert thresh_l2 == 15
    assert wait_l2 >= 20

    is_due_l3, wait_l3, thresh_l3 = db.check_retriage_due(l3_enc, now=now)
    assert is_due_l3 is False
    assert thresh_l3 == 60

    # Metrics must track exactly 1 retriage due patient
    metrics = db.get_metrics()
    assert metrics["retriage_due_count"] == 1

    # Simulate clinical deterioration re-triage on L3 patient (SpO2 drops from 97% to 84% -> forces L1 Resuscitation)
    updated = db.re_triage_encounter(
        encounter_id="ENC-RT-L3",
        new_vitals={"spo2": 84.0, "pulse": 135},
        complaint_update="Patient suddenly became pale and unresponsive"
    )
    assert updated is not None
    assert updated["tier"] == 1  # Escalated to L1 Resuscitation
    assert updated["is_escalated"] is True
    assert "last_retriage_at" in updated
    assert any("Re-Triage" in a.get("action", "") for a in updated.get("audit", []))
