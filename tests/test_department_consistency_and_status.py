"""CarePriority Gov - Department Consistency & Dashboard Status Regression Suite
=============================================================================
Regression tests verifying:
1. All 13 canonical hospital departments are present and consistent across configs, models, and rules.
2. Department rules correctly route to Dental, Dermatology, Ophthalmology, Psychiatry, General Surgery, etc.
3. Queue filters correctly filter across the 13 canonical departments.
4. Invalid department labels in nurse overrides are rejected safely.
5. Emergency escalation rules take precedence over ML proposals.
6. Uncertain/incomplete cases trigger Safety Abstention to Needs Nurse Review.
7. Nurse overrides require >=5 char rationale, secondary escalation downgrade confirmation, and preserve audit trails.
8. Dashboard status messages truthfully distinguish loaded ML models from placeholder modes.
"""

import os
import copy
from datetime import datetime
import pytest

from rules_config import DEPARTMENTS, DEPARTMENT_RULES
from triage_engine import (
    recommend_department,
    run_triage,
    validate_nurse_override,
    check_model_status,
    load_ml_models
)
import db

CANONICAL_13_DEPTS = [
    "General Medicine",
    "Paediatrics",
    "Obstetrics & Gynaecology",
    "General Surgery",
    "Orthopaedics",
    "ENT",
    "Ophthalmology",
    "Dermatology",
    "Psychiatry",
    "Dental",
    "Cardiology",
    "Pulmonology/Chest & TB",
    "Emergency/Casualty"
]


def test_all_13_canonical_departments_present_and_exact():
    """DEPARTMENTS in rules_config.py must match the exact 13 canonical organizer departments."""
    assert len(DEPARTMENTS) == 13
    assert sorted(DEPARTMENTS) == sorted(CANONICAL_13_DEPTS)

    # Ensure no legacy variants remain in DEPARTMENTS list
    assert "Pediatrics" not in DEPARTMENTS
    assert "Orthopedics" not in DEPARTMENTS
    assert "Respiratory / Pulmonology" not in DEPARTMENTS
    assert "Emergency / Resuscitation" not in DEPARTMENTS


def test_department_rules_cover_canonical_and_legacy_aliases():
    """DEPARTMENT_RULES must have entries for all canonical departments plus backward-compatibility aliases."""
    for dept in CANONICAL_13_DEPTS:
        assert dept in DEPARTMENT_RULES, f"Missing rules for canonical department: {dept}"

    # Legacy aliases must also resolve cleanly without KeyError
    assert "Pediatrics" in DEPARTMENT_RULES
    assert "Orthopedics" in DEPARTMENT_RULES
    assert "Respiratory / Pulmonology" in DEPARTMENT_RULES
    assert "Emergency / Resuscitation" in DEPARTMENT_RULES


def test_recommend_department_routes_specialties_accurately():
    """Rule-based department recommendation must route complaints to appropriate canonical departments."""
    # Emergency / Casualty for Level 1
    assert recommend_department("unconscious patient", age=50, gender="Male", tier=1) == "Emergency/Casualty"

    # Dental
    res_dental = recommend_department("severe toothache and wisdom tooth swelling", age=30, gender="Male", tier=3)
    assert res_dental == "Dental"

    # Dermatology
    res_derm = recommend_department("itchy skin rash and severe eczema on arms", age=25, gender="Female", tier=4)
    assert res_derm == "Dermatology"

    # Ophthalmology
    res_eye = recommend_department("blurry vision and red eye with cataract", age=65, gender="Male", tier=4)
    assert res_eye == "Ophthalmology"

    # Psychiatry
    res_psych = recommend_department("severe depression, insomnia and panic attacks", age=40, gender="Female", tier=3)
    assert res_psych == "Psychiatry"

    # General Surgery
    res_surg = recommend_department("deep dog bite on leg with open bleeding wound", age=35, gender="Male", tier=3)
    assert res_surg == "General Surgery"

    # Paediatrics (age <= 14)
    res_paed = recommend_department("mild cough and cold", age=6, gender="Female", tier=4)
    assert res_paed == "Paediatrics"

    # Obstetrics & Gynaecology (female with obstetric symptoms)
    res_obgyn = recommend_department("pregnancy pain and vaginal discharge", age=28, gender="Female", tier=3)
    assert res_obgyn == "Obstetrics & Gynaecology"

    # Orthopaedics
    res_ortho = recommend_department("severe knee pain and suspected bone fracture after fall", age=45, gender="Male", tier=3)
    assert res_ortho == "Orthopaedics"

    # Pulmonology/Chest & TB
    res_pulm = recommend_department("chronic cough for 3 weeks and tuberculosis checkup", age=50, gender="Male", tier=3)
    assert res_pulm == "Pulmonology/Chest & TB"


def test_department_filters_and_queue_include_canonical_departments():
    """Queue retrieval with canonical department filters must function correctly."""
    now = datetime.now()
    records = [
        {
            "encounter_id": "TEST-DEPT-DENTAL",
            "patient_name": "Dental Patient",
            "created_at": now.isoformat(),
            "tier": 4,
            "department": "Dental",
            "status": "Waiting"
        },
        {
            "encounter_id": "TEST-DEPT-DERM",
            "patient_name": "Dermatology Patient",
            "created_at": now.isoformat(),
            "tier": 4,
            "department": "Dermatology",
            "status": "Waiting"
        },
        {
            "encounter_id": "TEST-DEPT-SURG",
            "patient_name": "Surgery Patient",
            "created_at": now.isoformat(),
            "tier": 3,
            "department": "General Surgery",
            "status": "Waiting"
        }
    ]
    for r in records:
        db.insert_encounter(r)

    dental_queue = db.get_queue(department_filter="Dental")
    assert any(p["encounter_id"] == "TEST-DEPT-DENTAL" for p in dental_queue)
    assert not any(p["encounter_id"] == "TEST-DEPT-DERM" for p in dental_queue)

    derm_queue = db.get_queue(department_filter="Dermatology")
    assert any(p["encounter_id"] == "TEST-DEPT-DERM" for p in derm_queue)

    surg_queue = db.get_queue(department_filter="General Surgery")
    assert any(p["encounter_id"] == "TEST-DEPT-SURG" for p in surg_queue)


def test_invalid_department_labels_handled_safely_in_nurse_override():
    """Nurse override must reject bogus department strings and accept canonical ones."""
    # Invalid department
    ok, err = validate_nurse_override(
        current_tier=3,
        new_tier=3,
        current_dept="General Medicine",
        new_dept="Astrology Clinic",
        reason="Patient requests astrology consultation",
        is_escalated=False
    )
    assert ok is False
    assert "Unknown department" in err

    # Valid canonical department
    ok_valid, err_valid = validate_nurse_override(
        current_tier=3,
        new_tier=3,
        current_dept="General Medicine",
        new_dept="Dental",
        reason="Found acute dental abscess on physical exam",
        is_escalated=False
    )
    assert ok_valid is True
    assert err_valid is None

    # Valid legacy alias normalized
    ok_alias, err_alias = validate_nurse_override(
        current_tier=3,
        new_tier=3,
        current_dept="General Medicine",
        new_dept="Pediatrics",
        reason="Patient is 10 years old, transferring to paediatrics",
        is_escalated=False
    )
    assert ok_alias is True


def test_emergency_escalation_rules_take_precedence_over_ml_predictions():
    """Hard-coded safety escalations must force Level 1 or Level 2 regardless of ML model or mild symptoms."""
    # Hypoxia SpO2 84% must force Level 1
    res_l1 = run_triage(
        patient_id="TEST-SAFETY-L1",
        patient_name="Hypoxic Patient",
        age=65,
        gender="Male",
        spo2=84.0,
        bp_systolic=110,
        bp_diastolic=70,
        pulse=110,
        temperature=98.6,
        complaint="Mild tiredness"
    )
    assert res_l1["tier"] == 1
    assert res_l1["is_escalated"] is True
    assert res_l1["department"] == "Emergency/Casualty"

    # Hypoxia SpO2 90% must force at least Level 2
    res_l2 = run_triage(
        patient_id="TEST-SAFETY-L2",
        patient_name="Moderate Hypoxia Patient",
        age=50,
        gender="Female",
        spo2=90.0,
        bp_systolic=120,
        bp_diastolic=80,
        pulse=85,
        temperature=98.6,
        complaint="Routine checkup"
    )
    assert res_l2["tier"] in [1, 2]
    assert res_l2["is_escalated"] is True


def test_uncertain_or_incomplete_cases_referred_for_nurse_review():
    """Uninformative or missing intakes must trigger Safety Abstention and never default to Level 5."""
    # Completely empty vitals and complaint
    res_empty = run_triage(
        patient_id="TEST-ABSTAIN-01",
        patient_name="Incomplete Walkin",
        age=40,
        gender="Other",
        spo2=None,
        bp_systolic=None,
        bp_diastolic=None,
        pulse=None,
        temperature=None,
        complaint=""
    )
    assert res_empty["abstained"] is True
    assert res_empty["tier"] is None
    assert res_empty["tier_name"] == "Needs Nurse Review"
    assert res_empty["department"] == "Triage Review Desk"

    # Unclear short token without vitals
    res_token = run_triage(
        patient_id="TEST-ABSTAIN-02",
        patient_name="Token Patient",
        age=30,
        gender="Male",
        spo2=None,
        bp_systolic=None,
        bp_diastolic=None,
        pulse=None,
        temperature=None,
        complaint=".."
    )
    assert res_token["abstained"] is True
    assert res_token["tier"] is None


def test_nurse_override_requires_rationale_and_preserves_audit_logging():
    """Nurse override must require minimum rationale length and record an immutable audit entry."""
    # Rationale < 5 chars rejected
    ok_short, err_short = validate_nurse_override(
        current_tier=4,
        new_tier=2,
        current_dept="General Medicine",
        new_dept="Cardiology",
        reason="abc",
        is_escalated=False
    )
    assert ok_short is False
    assert "minimum 5 characters" in err_short

    # Hard escalation downgrade without checkbox rejected
    ok_down, err_down = validate_nurse_override(
        current_tier=2,
        new_tier=4,
        current_dept="Emergency/Casualty",
        new_dept="General Medicine",
        reason="Patient looks fine to me clinically",
        is_escalated=True,
        confirm_escalation_downgrade=False
    )
    assert ok_down is False
    assert "Downgrading requires an explicit secondary clinician confirmation" in err_down

    # Re-triage deterioration preserves audit log
    enc_id = "TEST-AUDIT-RETRIAGE"
    db.insert_encounter({
        "encounter_id": enc_id,
        "patient_name": "Audit Test Patient",
        "created_at": datetime.now().isoformat(),
        "tier": 4,
        "tier_name": "Less Urgent",
        "department": "General Medicine",
        "status": "Waiting",
        "complaint_original": "Mild fever",
        "vitals": {"spo2": 97.0, "bp_systolic": 120, "pulse": 75, "temperature": 99.0},
        "audit": []
    })

    # Patient deteriorates: SpO2 drops to 84%
    updated = db.re_triage_encounter(
        encounter_id=enc_id,
        new_vitals={"spo2": 84.0},
        complaint_update="Patient started gasping for breath",
        actor="ICU Nurse Sharma"
    )
    assert updated is not None
    assert updated["tier"] in [1, 2]
    assert len(updated["audit"]) == 1
    latest_audit = updated["audit"][-1]
    assert "Deterioration Escalated" in latest_audit["action"]
    assert latest_audit["actor"] == "ICU Nurse Sharma"
    assert latest_audit["old_tier"] == 4


def test_dashboard_status_component_truthful_representation():
    """Status helper must accurately reflect model and data availability without false claims."""
    status = check_model_status()
    # If model files exist on disk, has_models must be True
    if os.path.exists("models/triage_pipeline.joblib") and os.path.exists("models/dept_pipeline.joblib"):
        assert status["has_models"] is True
        assert "ML models available" in status["status_msg"]

    # Trained models must cover 13 canonical departments
    t_pipe, d_pipe, meta = load_ml_models()
    if d_pipe is not None and hasattr(d_pipe, "classes_"):
        assert len(d_pipe.classes_) == 13
        assert sorted(d_pipe.classes_) == sorted(CANONICAL_13_DEPTS)
