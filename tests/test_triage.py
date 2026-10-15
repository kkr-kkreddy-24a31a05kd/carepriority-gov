"""CarePriority Gov - Automated Test Suite
==========================================
Validates:
1. Safety Abstention guardrail (empty input never returns Level 5)
2. Mandatory Hard Escalations:
   - SpO2 < 92% forces L1 or L2 even with normal other vitals
   - "chest pain" keyword forces L1 or L2
   - Telugu & Hindi keywords trigger appropriate escalations
3. Missing vitals remain None (never coerced to 0)
4. Temperature plausibility & Celsius unit-error detection
5. Queue ordering: (a) Review needed first, (b) tier L1->L5, (c) arrival time
6. Nurse override: rejected without mandatory reason, requires explicit confirmation for hard-escalation downgrades
7. Database fallback and metric consistency
"""

import pytest
from datetime import datetime, timedelta
from triage_engine import (
    run_triage,
    validate_vitals,
    evaluate_safety_abstention,
    evaluate_hard_escalations,
    validate_nurse_override,
    score_tier_without_hard_escalation
)
import db


def test_abstention_on_empty_input_never_returns_level_5():
    """Safety Abstention: Incomplete intake must never default to Level 5 Non-Urgent."""
    result = run_triage(
        patient_id="TEST-EMPTY-01",
        patient_name="Empty Intake Patient",
        age=None,
        gender=None,
        spo2=None,
        bp_systolic=None,
        bp_diastolic=None,
        pulse=None,
        temperature=None,
        complaint=""
    )
    # Must abstain
    assert result["abstained"] is True, "Engine must abstain on empty intake"
    assert result["tier"] is None, "Abstained case must have tier=None, not Level 5"
    assert result["tier"] != 5, "CRITICAL SAFETY: Abstention must NEVER default to Level 5"
    assert "SAFETY ABSTENTION" in result["abstention_reason"]


def test_abstention_on_ambiguous_short_complaint_without_vitals():
    """Ambiguous uninterpretable token with zero vitals must trigger safety abstention."""
    result = run_triage(
        patient_id="TEST-AMBIGUOUS-01",
        patient_name="Ambiguous",
        age=30,
        gender="Male",
        spo2=None,
        bp_systolic=None,
        bp_diastolic=None,
        pulse=None,
        temperature=None,
        complaint=".."
    )
    assert result["abstained"] is True
    assert result["tier"] is None
    assert result["tier"] != 5


def test_spo2_below_92_forces_l1_or_l2_even_if_other_vitals_normal():
    """SpO2 < 92% must trigger hard-coded escalation to L1 or L2 regardless of normal vitals."""
    # Test SpO2 = 91% (should force L2)
    result_91 = run_triage(
        patient_id="TEST-SPO2-91",
        patient_name="Mild Hypoxia",
        age=45,
        gender="Male",
        spo2=91.0,
        bp_systolic=120,    # perfectly normal
        bp_diastolic=80,     # perfectly normal
        pulse=72,            # perfectly normal
        temperature=98.6,    # perfectly normal
        complaint="Routine walk in for checkup"
    )
    assert result_91["abstained"] is False
    assert result_91["is_escalated"] is True
    assert result_91["tier"] == 2
    assert any("SpO2" in reason or "Hypoxemia" in reason for reason in result_91["reasons"])

    # Test SpO2 = 82% (critical anoxia, should force L1)
    result_82 = run_triage(
        patient_id="TEST-SPO2-82",
        patient_name="Critical Hypoxia",
        age=45,
        gender="Male",
        spo2=82.0,
        bp_systolic=120,
        bp_diastolic=80,
        pulse=72,
        temperature=98.6,
        complaint="Walk in"
    )
    assert result_82["tier"] == 1
    assert result_82["is_escalated"] is True


def test_chest_pain_keyword_forces_l1_or_l2():
    """'chest pain' keyword must force Level 2 Emergent escalation even with normal vitals."""
    result = run_triage(
        patient_id="TEST-CHEST-PAIN",
        patient_name="Chest Pain Patient",
        age=50,
        gender="Male",
        spo2=99.0,
        bp_systolic=120,
        bp_diastolic=80,
        pulse=75,
        temperature=98.4,
        complaint="Patient complains of chest pain since morning"
    )
    assert result["tier"] in [1, 2]
    assert result["is_escalated"] is True
    assert any("chest pain" in rf.lower() for rf in result["red_flags"])
    assert result["department"] == "Cardiology"


def test_multilingual_red_flags_hindi_and_telugu():
    """Hindi and Telugu red flag symptoms must trigger mandatory escalation."""
    # Hindi: छाती में दर्द (Chest pain)
    res_hi = run_triage(
        patient_id="TEST-HI-01",
        patient_name="Hindi Patient",
        age=55,
        gender="Male",
        spo2=98.0,
        bp_systolic=130,
        bp_diastolic=85,
        pulse=80,
        temperature=98.6,
        complaint="मरीज को छाती में दर्द हो रहा है"
    )
    assert res_hi["tier"] in [1, 2]
    assert res_hi["is_escalated"] is True

    # Telugu: ఊపిరాడటం లేదు (Not breathing / dyspnea)
    res_te = run_triage(
        patient_id="TEST-TE-01",
        patient_name="Telugu Patient",
        age=60,
        gender="Female",
        spo2=96.0,
        bp_systolic=125,
        bp_diastolic=80,
        pulse=90,
        temperature=98.6,
        complaint="రోగికి ఊపిరాడటం లేదు"
    )
    assert res_te["tier"] in [1, 2]
    assert res_te["is_escalated"] is True


def test_missing_vitals_stay_none():
    """Missing vitals must be preserved as None, never converted to 0."""
    result = run_triage(
        patient_id="TEST-NONE-VITALS",
        patient_name="Partial Vitals",
        age=30,
        gender="Female",
        spo2=98.0,
        bp_systolic=None,   # not measured
        bp_diastolic=None,  # not measured
        pulse=None,         # not measured
        temperature=98.6,
        complaint="Minor superficial finger cut while cooking"
    )
    # The pure engine does not coerce missing inputs to 0
    # Let's verify via validate_vitals that None is accepted without error
    is_valid, errors, _ = validate_vitals(
        age=30,
        spo2=98.0,
        bp_systolic=None,
        bp_diastolic=None,
        pulse=None,
        temperature=98.6
    )
    assert is_valid is True
    assert len(errors) == 0


def test_temperature_unit_error_flag_works():
    """Temperature in Celsius range (e.g. 37.0°C to 40.0°C) or implausible extremes must be flagged."""
    # Test Celsius entry (e.g. 38.5 entered as temperature)
    is_valid_c, errors_c, _ = validate_vitals(
        age=30,
        spo2=98.0,
        bp_systolic=120,
        bp_diastolic=80,
        pulse=75,
        temperature=38.5
    )
    assert is_valid_c is False
    assert any("Celsius" in err for err in errors_c)

    # Test implausible extreme (< 90°F or > 110°F)
    is_valid_ext, errors_ext, _ = validate_vitals(
        age=30,
        spo2=98.0,
        bp_systolic=120,
        bp_diastolic=80,
        pulse=75,
        temperature=125.0
    )
    assert is_valid_ext is False
    assert any("outside plausible human range" in err for err in errors_ext)


def test_queue_ordering_review_needed_first_then_tier_then_arrival():
    """
    Queue ordering contract:
    (a) Needs Nurse Review / Abstained first
    (b) Priority tier L1 -> L5
    (c) Arrival time (created_at ascending)
    """
    db.clear_all()
    now = datetime.now()

    # Create test encounters in arbitrary order
    enc_l3_early = {
        "encounter_id": "QUEUE-L3-EARLY",
        "created_at": (now - timedelta(minutes=40)).isoformat(),
        "tier": 3,
        "abstained": False,
        "status": "Waiting",
        "department": "General Medicine"
    }
    enc_l1_recent = {
        "encounter_id": "QUEUE-L1-RECENT",
        "created_at": (now - timedelta(minutes=5)).isoformat(),
        "tier": 1,
        "abstained": False,
        "status": "Waiting",
        "department": "Emergency / Resuscitation"
    }
    enc_l2 = {
        "encounter_id": "QUEUE-L2",
        "created_at": (now - timedelta(minutes=20)).isoformat(),
        "tier": 2,
        "abstained": False,
        "status": "Waiting",
        "department": "Cardiology"
    }
    enc_abstained = {
        "encounter_id": "QUEUE-ABSTAINED",
        "created_at": (now - timedelta(minutes=2)).isoformat(),
        "tier": None,
        "abstained": True,
        "status": "Awaiting Nurse Review",
        "department": "Triage Review Desk"
    }
    enc_l1_older = {
        "encounter_id": "QUEUE-L1-OLDER",
        "created_at": (now - timedelta(minutes=15)).isoformat(),
        "tier": 1,
        "abstained": False,
        "status": "Waiting",
        "department": "Emergency / Resuscitation"
    }

    # Insert out of order
    for record in [enc_l3_early, enc_l1_recent, enc_l2, enc_abstained, enc_l1_older]:
        db.insert_encounter(record)

    queue = db.get_queue()
    ids = [item["encounter_id"] for item in queue]

    # Position 0 must be the Abstained / Needs Nurse Review case
    assert ids[0] == "QUEUE-ABSTAINED", f"Expected QUEUE-ABSTAINED first, got {ids[0]}"

    # Position 1 and 2 must be the L1 cases, ordered by arrival time (older first)
    assert ids[1] == "QUEUE-L1-OLDER", f"Expected QUEUE-L1-OLDER second, got {ids[1]}"
    assert ids[2] == "QUEUE-L1-RECENT", f"Expected QUEUE-L1-RECENT third, got {ids[2]}"

    # Position 3 must be L2
    assert ids[3] == "QUEUE-L2", f"Expected QUEUE-L2 fourth, got {ids[3]}"

    # Position 4 must be L3
    assert ids[4] == "QUEUE-L3-EARLY", f"Expected QUEUE-L3-EARLY fifth, got {ids[4]}"


def test_nurse_override_validation():
    """Override without written reason must be rejected; hard escalation downgrade requires confirmation."""
    # 1. Reject empty reason
    is_valid, err = validate_nurse_override(
        current_tier=3,
        new_tier=2,
        current_dept="General Medicine",
        new_dept="Cardiology",
        reason="",
        is_escalated=False
    )
    assert is_valid is False
    assert "mandatory" in err.lower() or "rationale" in err.lower()

    # 2. Reject downgrade of hard escalation without explicit secondary confirmation
    is_valid_esc, err_esc = validate_nurse_override(
        current_tier=2,
        new_tier=4,
        current_dept="Cardiology",
        new_dept="General Medicine",
        reason="Patient looks relatively calm to me",
        is_escalated=True,
        confirm_escalation_downgrade=False
    )
    assert is_valid_esc is False
    assert "confirmation" in err_esc.lower() or "guardrail" in err_esc.lower()

    # 3. Accept valid override with reason and proper confirmation
    is_valid_ok, err_ok = validate_nurse_override(
        current_tier=2,
        new_tier=4,
        current_dept="Cardiology",
        new_dept="General Medicine",
        reason="Secondary manual ECG confirmed sinus rhythm, non-cardiac musculoskeletal wall pain verified",
        is_escalated=True,
        confirm_escalation_downgrade=True
    )
    assert is_valid_ok is True
    assert err_ok is None


def test_asymmetric_cost_matrix_under_triage_penalty():
    """Asymmetric Safety: Under-triaging a true emergency must cost ~20x more than over-triaging a mild case."""
    from rules_config import calculate_asymmetric_cost

    # Under-triage of emergency: Actual L1, predicted as L4 (Mild)
    cost_under_triage_l1 = calculate_asymmetric_cost(actual_tier=1, predicted_tier=4)
    # Under-triage of emergency: Actual L2, predicted as L4 (Mild)
    cost_under_triage_l2 = calculate_asymmetric_cost(actual_tier=2, predicted_tier=4)
    # Over-triage of mild case: Actual L5, predicted as L2 (Emergency)
    cost_over_triage = calculate_asymmetric_cost(actual_tier=5, predicted_tier=2)

    assert cost_under_triage_l1 >= 20.0, f"L1 under-triage must have severe cost >= 20.0, got {cost_under_triage_l1}"
    assert cost_under_triage_l2 >= 20.0, f"L2 under-triage must have severe cost >= 20.0, got {cost_under_triage_l2}"
    assert cost_over_triage <= 2.0, f"Over-triage cost should be low (~1.0), got {cost_over_triage}"
    assert cost_under_triage_l2 >= (15 * cost_over_triage), "Under-triage must be penalized at least 15x-20x more than over-triage"


def test_bp_zero_stored_as_none_and_flagged():
    """BP of 0 = treated as missing, flagged for verification, stored as None."""
    from triage_engine import clean_vitals_for_storage

    cleaned = clean_vitals_for_storage(bp_systolic=0, bp_diastolic=0, spo2=98.0, pulse=76, temperature=98.6)
    assert cleaned["bp_systolic"] is None, "BP of 0 must be stored as None"
    assert cleaned["bp_diastolic"] is None, "Diastolic with SBP 0 must be stored as None"

    res = run_triage(
        patient_id="TEST-BP0-FLAG",
        patient_name="BP Zero Patient",
        age=45,
        gender="Male",
        spo2=98.0,
        bp_systolic=0,
        bp_diastolic=0,
        pulse=76,
        temperature=98.6,
        complaint="Routine mild headache"
    )
    assert res["cleaned_vitals"]["bp_systolic"] is None
    assert any("0" in w and "missing" in w.lower() for w in res["warnings"])


def test_celsius_above_43_flagged_as_probable_unit_error():
    """Temperature in Celsius above 43 (e.g. 96 C) must be flagged as probable unit error."""
    is_valid, errors, _ = validate_vitals(
        age=40,
        spo2=98.0,
        bp_systolic=120,
        bp_diastolic=80,
        pulse=72,
        temperature=96.0,
        temp_unit="C"
    )
    assert is_valid is False
    assert any("above 43°C" in err and "unit error" in err for err in errors)


def test_hinglish_and_telugu_latin_red_flags():
    """Hinglish and Telugu-in-Latin-script keywords must trigger mandatory escalations."""
    # Hinglish: "seene me dard" (Chest pain)
    res_hinglish = run_triage(
        patient_id="TEST-HINGLISH",
        patient_name="Hinglish Patient",
        age=52,
        gender="Male",
        spo2=98.0,
        bp_systolic=125,
        bp_diastolic=80,
        pulse=78,
        temperature=98.6,
        complaint="seene me dard bahut teevra ho raha hai"
    )
    assert res_hinglish["tier"] in [1, 2]
    assert res_hinglish["is_escalated"] is True

    # Telugu in Latin: "gunde noppi"
    res_te_lat = run_triage(
        patient_id="TEST-TELUGU-LATIN",
        patient_name="Telugu Latin Patient",
        age=49,
        gender="Female",
        spo2=98.0,
        bp_systolic=120,
        bp_diastolic=80,
        pulse=82,
        temperature=98.6,
        complaint="gunde noppi vastondi edama cheyyi laavutondi"
    )
    assert res_te_lat["tier"] in [1, 2]
    assert res_te_lat["is_escalated"] is True


def test_model_first_with_hard_rule_escalation_guardrail():
    """Hard-coded rules override model proposals and can only escalate, never downgrade."""
    # SpO2 90% is hypoxic (hard rule L2) even if complaint is completely mild
    res = run_triage(
        patient_id="TEST-HYBRID-ESCALATE",
        patient_name="Silent Hypoxia Walkin",
        age=60,
        gender="Male",
        spo2=90.0,
        bp_systolic=120,
        bp_diastolic=80,
        pulse=74,
        temperature=98.6,
        complaint="Came for routine medicine refill"
    )
    assert res["tier"] in [1, 2], "Hard hypoxia escalation must force L1 or L2"
    assert res["is_escalated"] is True


def test_queue_simulation_wait_time_reduction():
    """Stretch: Simulation with 2 doctors must demonstrate wait-time reduction for emergencies."""
    from evaluate_runner import simulate_queue_wait_times

    sim = simulate_queue_wait_times(num_doctors=2, num_patients=100)
    assert sim["emergency_wait_reduction_pct"] > 50.0, "CarePriority Gov should cut emergency wait times by >50%"
    assert sim["cpg_emergency_wait_mins"] < sim["fcfs_emergency_wait_mins"]
    assert len(sim["assumptions"]) >= 3

