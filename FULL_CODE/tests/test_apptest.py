"""CarePriority Gov - Streamlit AppTest Suite
==========================================
Verifies that all 7 navigation pages render without unhandled exceptions
and asserts that no markdown element contains leaked HTML code blocks
(lines starting with 4+ spaces followed by '<div' or '<span').

Submits sample patients (L1, L3, abstained) first, then validates each page,
including deep inspections of Encounter Detail View for L1, L3, and abstained records.
"""

import os
import re
from datetime import datetime
import pytest
from streamlit.testing.v1 import AppTest

import db

APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))

PAGES = [
    "📊 Command Center",
    "⚡ Patient Intake (<60s)",
    "📋 Live Doctor OPD Queue",
    "📺 Public Status Display",
    "🔍 Encounter Detail View",
    "🛡️ Safety Rules & Thresholds",
    "🧪 Live Evaluation & Benchmarks"
]

SAMPLE_PATIENTS = [
    {
        "encounter_id": "ENC-APPTEST-L1",
        "created_at": datetime.now().isoformat(),
        "patient_name": "Sunita Devi (L1 Critical)",
        "age": 63,
        "gender": "Female",
        "department": "Emergency Medicine",
        "tier": 1,
        "tier_name": "Resuscitation",
        "abstained": False,
        "abstention_reason": None,
        "is_escalated": True,
        "rule_triggered": "SpO2 Critical (< 85%)",
        "reasons": ["SpO2 critical (< 85%): 82.0%", "Emergent vital sign escalation"],
        "red_flags": ["chest pain", "severe breathlessness"],
        "complaint_original": "Crushing chest pain radiating to back and severe breathlessness",
        "vitals": {"spo2": 82.0, "bp_systolic": 78, "bp_diastolic": 48, "pulse": 136, "temperature": 99.1},
        "model_tier": 1,
        "model_confidence": 0.95,
        "model_probs": [0.86, 0.09, 0.03, 0.01, 0.01],
        "status": "Waiting",
        "audit": [
            {
                "timestamp": datetime.now().isoformat(),
                "actor": "Triage Nurse",
                "action": "Intake Submission",
                "tier": 1,
                "department": "Emergency Medicine",
                "reason": "Immediate resuscitation needed"
            }
        ]
    },
    {
        "encounter_id": "ENC-APPTEST-L3",
        "created_at": datetime.now().isoformat(),
        "patient_name": "Rajesh Kumar (L3 Urgent)",
        "age": 36,
        "gender": "Male",
        "department": "General Medicine",
        "tier": 3,
        "tier_name": "Urgent",
        "abstained": False,
        "abstention_reason": None,
        "is_escalated": False,
        "rule_triggered": None,
        "reasons": ["Acute persistent abdominal pain"],
        "red_flags": [],
        "complaint_original": "Severe abdominal pain for 5 hours",
        "vitals": {"spo2": 98.0, "bp_systolic": 124, "bp_diastolic": 80, "pulse": 84, "temperature": 98.8},
        "model_tier": 3,
        "model_confidence": 0.82,
        "model_probs": [0.02, 0.07, 0.74, 0.11, 0.06],
        "status": "Waiting",
        "audit": [
            {
                "timestamp": datetime.now().isoformat(),
                "actor": "Triage Nurse",
                "action": "Intake Submission",
                "tier": 3,
                "department": "General Medicine",
                "reason": "Standard triage intake"
            }
        ]
    },
    {
        "encounter_id": "ENC-APPTEST-ABSTAIN",
        "created_at": datetime.now().isoformat(),
        "patient_name": "Mohan Lal (Abstained)",
        "age": 48,
        "gender": "Male",
        "department": "General Medicine",
        "tier": None,
        "tier_name": "Needs Nurse Review",
        "abstained": True,
        "abstention_reason": "Ambiguous short complaint with unmeasured clinical vitals",
        "is_escalated": False,
        "rule_triggered": None,
        "reasons": ["Safety Abstention: Clinical vitals missing and complaint ambiguous"],
        "red_flags": [],
        "complaint_original": "Feeling sick",
        "vitals": {"spo2": None, "bp_systolic": None, "bp_diastolic": None, "pulse": None, "temperature": None},
        "model_tier": None,
        "model_confidence": None,
        "model_probs": None,
        "status": "Awaiting Nurse Review",
        "audit": [
            {
                "timestamp": datetime.now().isoformat(),
                "actor": "Safety Guardrail",
                "action": "Safety Abstention",
                "tier": None,
                "department": "General Medicine",
                "reason": "Missing vitals and ambiguous complaint"
            }
        ]
    }
]


@pytest.fixture(scope="module", autouse=True)
def seed_sample_patients():
    """Seeds sample L1, L3, and abstained encounters prior to running AppTest."""
    for p in SAMPLE_PATIENTS:
        db.insert_encounter(p)
    return True


def assert_no_leaked_html_markdown(at: AppTest, page_context: str):
    """
    Asserts that no markdown element contains lines with 4+ spaces
    preceding '<div' or '<span' (which causes CommonMark to treat it as code).
    """
    assert not at.exception, f"Exception occurred in {page_context}: {at.exception}"
    for i, md in enumerate(at.markdown):
        content = md.value or ""
        for line_num, line in enumerate(content.splitlines(), 1):
            assert not re.match(r"^\s{4,}<(div|span)", line, re.IGNORECASE), (
                f"Leaked raw HTML code block in {page_context} (markdown #{i}, line {line_num}): {line!r}"
            )


@pytest.mark.parametrize("page_name", PAGES)
def test_page_renders_without_exception_and_no_leaked_html(page_name):
    """Verifies that each page in app.py renders cleanly without exceptions and without leaked HTML code blocks."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception, f"Initial boot failed before selecting {page_name}: {at.exception}"
    
    if at.sidebar.radio and len(at.sidebar.radio) > 0:
        at.sidebar.radio[0].set_value(page_name)
        at.run()
        assert not at.exception, f"Exception on page '{page_name}': {at.exception}"
        assert_no_leaked_html_markdown(at, f"Page '{page_name}'")


def test_intake_submission_and_post_submit_card_no_leaked_html():
    """Submits a patient via the intake portal form and validates that the post-submit assessment card has no leaked HTML."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    if at.sidebar.radio and len(at.sidebar.radio) > 0:
        at.sidebar.radio[0].set_value("⚡ Patient Intake (<60s)")
        at.run()
    
    # Fill in intake form inputs
    if at.text_input:
        for ti in at.text_input:
            if "name" in ti.label.lower():
                ti.set_value("Rani Devi")
            elif "complaint" in ti.label.lower():
                ti.set_value("Severe persistent fever and coughing blood")

    if at.number_input:
        for ni in at.number_input:
            if "age" in ni.label.lower():
                ni.set_value(55)
            elif "spo2" in ni.label.lower():
                ni.set_value(89.0)

    # Submit intake form
    if at.button and len(at.button) > 0:
        # Click the submit button inside the form
        for btn in at.button:
            if "triage assessment" in btn.label.lower():
                btn.click()
                at.run()
                break

    assert not at.exception, f"Exception after intake submission: {at.exception}"
    assert_no_leaked_html_markdown(at, "Post-Submit Patient Intake Card")


@pytest.mark.parametrize("sample_id", ["ENC-APPTEST-L1", "ENC-APPTEST-L3", "ENC-APPTEST-ABSTAIN"])
def test_encounter_detail_view_deep_html_check(sample_id):
    """Inspects Encounter Detail View for L1, L3, and Abstained encounters and verifies zero leaked HTML code blocks."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    if at.sidebar.radio and len(at.sidebar.radio) > 0:
        at.sidebar.radio[0].set_value("🔍 Encounter Detail View")
        at.run()

    # Find the matching option in the selectbox
    if at.selectbox and len(at.selectbox) > 0:
        target_opt = None
        for opt in at.selectbox[0].options:
            if sample_id in str(opt):
                target_opt = opt
                break
        
        if target_opt:
            at.selectbox[0].set_value(target_opt)
            at.run()

    assert not at.exception, f"Exception on Encounter Detail View for {sample_id}: {at.exception}"
    assert_no_leaked_html_markdown(at, f"Encounter Detail View ({sample_id})")
