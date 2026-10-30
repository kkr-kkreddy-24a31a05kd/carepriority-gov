"""Regression tests for emergency precedence and safe review routing."""

from pathlib import Path

import pytest

import evaluate_runner
import db
import triage_engine
from smart_ticket import render_smart_token_ticket
from triage_engine import run_triage, validate_vitals


def assess(**overrides):
    values = {
        "patient_id": "TEST-SAFETY-REGRESSION",
        "patient_name": "Test Patient",
        "age": 40,
        "gender": "Unknown",
        "spo2": 98.0,
        "bp_systolic": 120,
        "bp_diastolic": 80,
        "pulse": 75,
        "temperature": 98.6,
        "complaint": "Routine outpatient visit",
    }
    values.update(overrides)
    return run_triage(**values)


def hard_result(**overrides):
    values = {
        "spo2": 98.0,
        "bp_systolic": 120,
        "bp_diastolic": 80,
        "pulse": 75,
        "temperature": 98.6,
        "complaint": "Routine outpatient visit",
        "age": 40,
    }
    values.update(overrides)
    return triage_engine.evaluate_hard_escalations(**values)


def test_red_flag_emergency_precedes_missing_vitals_abstention():
    result = assess(
        spo2=None,
        bp_systolic=None,
        bp_diastolic=None,
        pulse=None,
        temperature=None,
        complaint="Patient is unresponsive",
    )

    assert result["abstained"] is False
    assert result["is_escalated"] is True
    assert result["tier"] == 1
    assert result["department"] == "Emergency/Casualty"
    assert result["red_flags"]


def test_emergency_rule_precedes_invalid_ancillary_input():
    result = assess(age=140, spo2=91.0)

    assert result["is_escalated"] is True
    assert result["abstained"] is False
    assert result["tier"] == 2
    assert any("Age must be a valid" in warning for warning in result["warnings"])


def test_model_can_escalate_but_never_downgrade_hard_rule(monkeypatch):
    monkeypatch.setattr(
        triage_engine,
        "predict_ml_triage",
        lambda **_: (1, 0.9, [0.8, 0.1, 0.05, 0.03, 0.02], "Cardiology"),
    )

    result = assess(spo2=91.0)

    assert result["is_escalated"] is True
    assert result["model_tier"] == 1
    assert result["tier"] == 1


def test_hard_rule_level_2_routes_to_emergency_casualty():
    result = assess(spo2=91.0, complaint="Toothache")

    assert result["is_escalated"] is True
    assert result["tier"] == 2
    assert result["department"] == "Emergency/Casualty"


@pytest.mark.parametrize(
    "complaint",
    [
        "New chest pain with sweating",
        "Chest pressure that may be cardiac",
        "Chest heaviness today",
        "Chest tightness",
    ],
)
def test_genuine_chest_discomfort_triggers_emergency_red_flag(complaint):
    tier, _, _, matches = hard_result(complaint=complaint)

    assert tier == 2
    assert matches


@pytest.mark.parametrize(
    "complaint",
    [
        "Chest pain clearly caused by an injury",
        "Chest pain when coughing",
        "Chest pain clearly caused by coughing",
        "Chest pain after meals",
        "Chest pain clearly caused by meals",
        "Denies chest pain",
        "No chest pain",
        "No chest pain is present",
    ],
)
def test_noncardiac_or_negated_chest_discomfort_does_not_trigger_chest_red_flag(complaint):
    tier, _, _, matches = hard_result(complaint=complaint)

    assert tier is None
    assert matches == []


def test_negated_chest_pain_does_not_suppress_another_genuine_red_flag():
    tier, _, _, matches = hard_result(
        complaint="No chest pain, but the patient is unresponsive"
    )

    assert tier == 1
    assert "unresponsive" in matches


@pytest.mark.parametrize("model_tier", [1, 2])
def test_ml_emergency_tiers_route_to_emergency_casualty(monkeypatch, model_tier):
    monkeypatch.setattr(
        triage_engine,
        "predict_ml_triage",
        lambda **_: (
            model_tier,
            0.9,
            [0.8, 0.1, 0.05, 0.03, 0.02],
            "Cardiology",
        ),
    )

    result = assess(complaint="Routine outpatient visit")

    assert result["abstained"] is False
    assert result["is_escalated"] is False
    assert result["tier"] == model_tier
    assert result["department"] == "Emergency/Casualty"


@pytest.mark.parametrize("model_tier", [3, 4, 5])
def test_ml_non_emergency_tiers_preserve_specialty_department(monkeypatch, model_tier):
    monkeypatch.setattr(
        triage_engine,
        "predict_ml_triage",
        lambda **_: (
            model_tier,
            0.9,
            [0.01, 0.01, 0.05, 0.03, 0.9],
            "Dental",
        ),
    )

    result = assess(complaint="Routine outpatient visit")

    assert result["tier"] == model_tier
    assert result["department"] == "Dental"


def test_invalid_vitals_without_emergency_route_to_nurse_review():
    result = assess(age=140)

    assert result["abstained"] is True
    assert result["tier"] is None
    assert result["department"] == "Triage Review Desk"
    assert "Vitals validation failed" in result["abstention_reason"]


def test_missing_complaint_with_multiple_usable_vitals_remains_assessable(monkeypatch):
    monkeypatch.setattr(
        triage_engine,
        "predict_ml_triage",
        lambda **_: (5, 0.9, [0.01, 0.01, 0.05, 0.03, 0.9], "General Medicine"),
    )
    result = assess(complaint="")

    assert result["abstained"] is False
    assert result["tier"] == 5


def test_complaint_without_primary_vitals_routes_to_nurse_review():
    result = assess(
        spo2=None,
        bp_systolic=None,
        bp_diastolic=None,
        pulse=None,
        temperature=98.6,
        complaint="Fever",
    )

    assert result["abstained"] is True
    assert result["tier"] is None


def test_zero_systolic_with_active_pulse_routes_to_nurse_review():
    result = assess(bp_systolic=0, bp_diastolic=0)

    assert result["abstained"] is True
    assert result["tier"] is None
    assert any("Contradictory hemodynamic record" in reason for reason in result["reasons"])


def test_valid_high_celsius_temperature_uses_primer_amber_tier(monkeypatch):
    monkeypatch.setattr(
        triage_engine,
        "predict_ml_triage",
        lambda **_: (5, 0.9, [0.01, 0.01, 0.05, 0.03, 0.9], "General Medicine"),
    )
    result = assess(temperature=41.0, temp_unit="C")

    assert result["abstained"] is False
    assert result["is_escalated"] is False
    assert result["tier"] == 3
    assert result["cleaned_vitals"]["temperature"] == pytest.approx(105.8)


def test_valid_normal_celsius_temperature_does_not_escalate(monkeypatch):
    monkeypatch.setattr(
        triage_engine,
        "predict_ml_triage",
        lambda **_: (4, 0.9, [0.01, 0.01, 0.05, 0.9, 0.03], "General Medicine"),
    )
    result = assess(temperature=37.0, temp_unit="C")

    assert result["abstained"] is False
    assert result["is_escalated"] is False
    assert result["tier"] == 4
    assert result["cleaned_vitals"]["temperature"] == pytest.approx(98.6)


def test_invalid_celsius_temperature_routes_to_nurse_review():
    result = assess(temperature=96.0, temp_unit="C")

    assert result["abstained"] is True
    assert result["tier"] is None
    assert result["is_escalated"] is False
    assert result["cleaned_vitals"]["temperature"] is None
    assert "30.0°C–45.0°C" in result["abstention_reason"]


def test_invalid_celsius_does_not_suppress_critical_spo2():
    result = assess(temperature=96.0, temp_unit="C", spo2=82.0)

    assert result["abstained"] is False
    assert result["is_escalated"] is True
    assert result["tier"] == 1
    assert result["cleaned_vitals"]["temperature"] is None


def test_invalid_celsius_does_not_suppress_critical_bp():
    result = assess(temperature=96.0, temp_unit="C", bp_systolic=60)

    assert result["abstained"] is False
    assert result["is_escalated"] is True
    assert result["tier"] == 1
    assert result["cleaned_vitals"]["temperature"] is None


def test_missing_temperature_preserves_existing_triage_behavior(monkeypatch):
    monkeypatch.setattr(
        triage_engine,
        "predict_ml_triage",
        lambda **_: (4, 0.9, [0.01, 0.01, 0.05, 0.9, 0.03], "General Medicine"),
    )
    result = assess(temperature=None)

    assert result["abstained"] is False
    assert result["is_escalated"] is False
    assert result["tier"] == 4
    assert result["cleaned_vitals"]["temperature"] is None


def test_valid_fahrenheit_high_temperature_uses_primer_amber_tier(monkeypatch):
    monkeypatch.setattr(
        triage_engine,
        "predict_ml_triage",
        lambda **_: (5, 0.9, [0.01, 0.01, 0.05, 0.03, 0.9], "General Medicine"),
    )
    result = assess(temperature=104.0)

    assert result["abstained"] is False
    assert result["is_escalated"] is False
    assert result["tier"] == 3
    assert result["cleaned_vitals"]["temperature"] == 104.0


def test_temperature_input_validation_preserves_invalid_case_for_review():
    valid, errors, _ = validate_vitals(
        age=40,
        spo2=98.0,
        bp_systolic=120,
        bp_diastolic=80,
        pulse=75,
        temperature=125.0,
    )

    assert valid is False
    assert any("outside the primer's usable range" in error for error in errors)


@pytest.mark.parametrize(
    ("age_months", "pulse", "expected_tier"),
    [
        (2, 180, None), (2, 181, 2),
        (3, 160, None), (3, 161, 2),
        (35, 160, None), (35, 161, 2),
        (36, 140, None), (36, 141, 2),
        (96, 120, None), (96, 121, 2),
        (180, 40, 2), (191, 40, 2),
        (192, 40, 2), (192, 41, None),
        (192, 130, None), (192, 131, 2),
    ],
)
def test_primer_age_specific_pulse_danger_zones(age_months, pulse, expected_tier):
    assert hard_result(age_months=age_months, pulse=pulse)[0] == expected_tier


@pytest.mark.parametrize(
    ("age_months", "systolic", "expected_tier"),
    [
        (0.5, 59, 1), (0.5, 60, None),
        (6, 69, 1), (6, 70, None),
        (24, 73, 1), (24, 74, None),
        (120, 79, 1), (120, 80, 2),
    ],
)
def test_primer_age_specific_pediatric_and_adult_hypotension(age_months, systolic, expected_tier):
    assert hard_result(age_months=age_months, bp_systolic=systolic)[0] == expected_tier


def test_primer_l1_child_bradycardia_and_low_oxygen_with_distress():
    assert hard_result(age_months=95, pulse=59)[0] == 1
    assert hard_result(age_months=96, pulse=59)[0] is None
    assert hard_result(spo2=89, complaint="struggling to breathe")[0] == 1
    assert hard_result(spo2=89, complaint="routine visit")[0] == 2


def test_primer_age_and_pregnancy_blood_pressure_thresholds():
    assert hard_result(age=10, bp_systolic=90)[0] == 2
    assert hard_result(age=9, bp_systolic=90)[0] is None
    assert hard_result(age=16, bp_systolic=220)[0] == 2
    assert hard_result(age=16, bp_systolic=219)[0] is None
    assert hard_result(age=40, bp_systolic=160)[0] is None
    assert hard_result(pregnant=True, bp_systolic=160)[0] == 2
    assert hard_result(pregnant=True, bp_diastolic=110)[0] == 2


def test_primer_infant_temperature_rules_and_red_flag_code():
    febrile = hard_result(age_months=2, temperature=100.4)
    hypothermic = hard_result(age_months=2, temperature=95.0)
    normal = hard_result(age_months=2, temperature=98.6)

    assert febrile[0] == 2
    assert "RF_HIGH_FEVER_INFANT" in febrile[3]
    assert hypothermic[0] == 2
    assert normal[0] is None


def test_primer_pregnancy_context_red_flags():
    bleeding = hard_result(
        pregnant=True,
        complaint="Vaginal bleeding during pregnancy",
    )
    eclampsia = hard_result(
        pregnant=True,
        pregnancy_stage="More than 20 weeks",
        complaint="Severe headache and blurred vision",
    )
    early_headache = hard_result(
        pregnant=True,
        pregnancy_stage="20 weeks or less",
        complaint="Severe headache and blurred vision",
    )
    postpartum = hard_result(
        recently_delivered=True,
        complaint="Blurred vision after delivery",
    )
    fetal_movement = hard_result(
        pregnant=True,
        pregnancy_stage="Third trimester",
        complaint="Reduced fetal movement",
    )

    assert bleeding[0] == 2 and "RF_PREGNANCY_BLEEDING" in bleeding[3]
    assert eclampsia[0] == 2 and "RF_ECLAMPSIA_SIGNS" in eclampsia[3]
    assert early_headache[0] is None
    assert postpartum[0] == 2 and "RF_ECLAMPSIA_SIGNS" in postpartum[3]
    assert fetal_movement[0] == 2 and "RF_REDUCED_FETAL_MOVEMENT" in fetal_movement[3]


@pytest.mark.parametrize(
    ("value", "unit", "valid"),
    [
        (30.0, "C", True), (45.0, "C", True), (45.1, "C", False),
        (86.0, "F", True), (113.0, "F", True), (113.1, "F", False),
    ],
)
def test_primer_temperature_plausibility_boundaries(value, unit, valid):
    result = validate_vitals(
        age=40,
        spo2=98,
        bp_systolic=120,
        bp_diastolic=80,
        pulse=75,
        temperature=value,
        temp_unit=unit,
    )
    assert result[0] is valid


def test_invalid_vitals_do_not_trigger_emergency_but_valid_independent_trigger_does():
    invalid_spo2 = assess(spo2=0)
    invalid_pulse = assess(pulse=0)
    independent_hypoxia = assess(spo2=91, pulse=0)

    assert invalid_spo2["abstained"] is True
    assert invalid_spo2["is_escalated"] is False
    assert invalid_pulse["abstained"] is True
    assert invalid_pulse["is_escalated"] is False
    assert independent_hypoxia["is_escalated"] is True
    assert independent_hypoxia["tier"] == 2


@pytest.mark.parametrize(
    ("complaint", "canonical_code", "expected_tier"),
    [
        ("Sudden facial droop began this morning", "RF_STROKE_SIGNS", 2),
        ("Sudden weakness in left arm", "RF_STROKE_SIGNS", 2),
        ("Patient cannot speak full sentences", "RF_RESP_DISTRESS", 2),
        ("Sick child has chest indrawing", "RF_RESP_DISTRESS", 2),
        ("Severe difficulty breathing", "RF_RESP_DISTRESS", 2),
        ("New confusion reported by family", "RF_ALTERED_SENSORIUM", 2),
        ("Had a seizure within the last few hours", "RF_SEIZURE", 2),
        ("Had a seizure 2 hours ago", "RF_SEIZURE", 2),
        ("Patient vomited blood", "RF_SEVERE_BLEEDING", 2),
        ("Unresponsive to voice", "RF_ALTERED_SENSORIUM", 1),
        ("Convulsing now", "RF_SEIZURE", 1),
        ("Wound bleeding will not stop", "RF_SEVERE_BLEEDING", 1),
        ("Bleeding that will not stop", "RF_SEVERE_BLEEDING", 1),
    ],
)
def test_stage1_canonical_red_flags_escalate_at_primer_tier(
    monkeypatch, complaint, canonical_code, expected_tier
):
    monkeypatch.setattr(
        triage_engine,
        "predict_ml_triage",
        lambda **_: (5, 0.99, [0.0, 0.0, 0.0, 0.0, 1.0], "General Medicine"),
    )

    result = assess(complaint=complaint)

    assert canonical_code in result["red_flags"]
    assert result["tier"] == expected_tier
    assert result["department"] == "Emergency/Casualty"


@pytest.mark.parametrize(
    ("complaint", "canonical_code"),
    [
        ("Chronic facial droop for years", "RF_STROKE_SIGNS"),
        ("Shortness of breath, but speaking comfortably", "RF_RESP_DISTRESS"),
        ("Longstanding confusion with no new confusion", "RF_ALTERED_SENSORIUM"),
        ("Seizure last year, none within the last few hours", "RF_SEIZURE"),
        ("Small blood streaks after coughing", "RF_SEVERE_BLEEDING"),
        ("Weakness in left arm without sudden onset", "RF_STROKE_SIGNS"),
        ("Sudden weakness without a face, arm, leg, or speech detail", "RF_STROKE_SIGNS"),
        ("Child has chest indrawing but is not sick", "RF_RESP_DISTRESS"),
        ("Ordinary shortness of breath", "RF_RESP_DISTRESS"),
        ("Mild difficulty breathing", "RF_RESP_DISTRESS"),
        ("Seizure 12 hours ago", "RF_SEIZURE"),
        ("No bleeding that will not stop", "RF_SEVERE_BLEEDING"),
        ("No sudden facial droop", "RF_STROKE_SIGNS"),
        ("Not struggling to breathe", "RF_RESP_DISTRESS"),
        ("No new confusion", "RF_ALTERED_SENSORIUM"),
        ("No seizure within the last few hours", "RF_SEIZURE"),
        ("No seizure 2 hours ago", "RF_SEIZURE"),
        ("No vomiting blood", "RF_SEVERE_BLEEDING"),
    ],
)
def test_stage1_canonical_red_flags_reject_nonqualifying_or_negated_symptoms(
    complaint, canonical_code
):
    forced_tier, _, _, red_flags = hard_result(complaint=complaint)

    assert canonical_code not in red_flags
    assert forced_tier is None


@pytest.mark.parametrize(
    ("complaint", "expected_codes", "rejected_codes"),
    [
        (
            "No slurred speech, but sudden face droop is present",
            {"RF_STROKE_SIGNS"},
            set(),
        ),
        (
            "No vomiting blood, but black tarry stools today",
            {"RF_SEVERE_BLEEDING"},
            set(),
        ),
        (
            "No seizure in recent hours, but new confusion is present",
            {"RF_ALTERED_SENSORIUM"},
            {"RF_SEIZURE"},
        ),
    ],
)
def test_stage1_negation_is_local_and_keeps_other_red_flags(
    complaint, expected_codes, rejected_codes
):
    forced_tier, _, _, red_flags = hard_result(complaint=complaint)

    assert expected_codes.issubset(set(red_flags))
    assert rejected_codes.isdisjoint(set(red_flags))
    assert forced_tier in (1, 2)


def test_stage1_canonical_red_flags_are_shown_in_explainability():
    from explainability import compute_factual_contributions

    contributions = compute_factual_contributions({
        "complaint": "Wound bleeding will not stop",
        "vitals": {
            "spo2": 98.0,
            "bp_systolic": 120,
            "bp_diastolic": 80,
            "pulse": 75,
            "temperature": 98.6,
        },
        "age": 40,
        "model_tier": 5,
    })

    canonical_factors = [
        factor for factor in contributions["red_flag_factors"]
        if factor["measured"] == "RF_SEVERE_BLEEDING"
    ]
    assert len(canonical_factors) == 1
    assert canonical_factors[0]["effect"] == "escalated to L1 (Resuscitation)"


def test_uninformative_complaint_abstention_uses_usable_vital_count():
    from triage_engine import evaluate_safety_abstention

    one_vital = evaluate_safety_abstention(
        complaint="doctor ko dikhana hai",
        spo2=98,
        bp_systolic=None,
        bp_diastolic=None,
        pulse=None,
        temperature=None,
    )
    multiple_vitals = evaluate_safety_abstention(
        complaint="doctor ko dikhana hai",
        spo2=98,
        bp_systolic=120,
        bp_diastolic=80,
        pulse=75,
        temperature=98.6,
    )

    assert one_vital[0] is True
    assert multiple_vitals[0] is False


@pytest.mark.parametrize(
    ("new_tier", "expected_department"),
    [(1, "Emergency/Casualty"), (2, "Emergency/Casualty")],
)
def test_recommend_department_routes_both_emergency_tiers(new_tier, expected_department):
    assert triage_engine.recommend_department(
        "toothache", age=40, gender="Unknown", tier=new_tier
    ) == expected_department


def test_primer_pediatric_department_boundary_and_specialty_exceptions():
    assert triage_engine.recommend_department("fever", 11, "Unknown", 4) == "Paediatrics"
    assert triage_engine.recommend_department("fever", 12, "Unknown", 4) == "General Medicine"
    assert triage_engine.recommend_department("toothache", 5, "Unknown", 4) == "Dental"
    assert triage_engine.recommend_department(
        "urine burning", 28, "Female", 3, pregnant=True
    ) == "Obstetrics & Gynaecology"


def test_amber_vitals_and_infant_level_five_guard_apply_over_ml(monkeypatch):
    monkeypatch.setattr(
        triage_engine,
        "predict_ml_triage",
        lambda **_: (5, 0.9, [0.01, 0.01, 0.05, 0.03, 0.9], "General Medicine"),
    )

    infant = assess(age=2 / 12, age_months=2, complaint="routine checkup")
    adult_pulse = assess(age=40, pulse=111)
    pregnant_bp = assess(pregnant=True, bp_systolic=140)
    adult_temp = assess(age=40, temperature=39.1, temp_unit="C")

    assert infant["tier"] == 4
    assert infant["department"] == "Paediatrics"
    assert adult_pulse["tier"] == 3
    assert pregnant_bp["tier"] == 3
    assert adult_temp["tier"] == 3


def test_retriage_preserves_infant_age_months_and_pregnancy_context():
    db.clear_all()
    db.insert_encounter({
        "encounter_id": "TEST-RETRIAGE-INFANT-MONTHS",
        "patient_name": "Test infant",
        "age": 2 / 12,
        "age_months": 2,
        "gender": "Unknown",
        "pregnant": False,
        "recently_delivered": False,
        "vitals": {
            "spo2": 98.0,
            "bp_systolic": 120,
            "bp_diastolic": 80,
            "pulse": 75,
            "temperature": 98.6,
        },
        "tier": 5,
        "abstained": False,
        "status": "Waiting",
        "audit": [],
    })

    updated = db.re_triage_encounter(
        "TEST-RETRIAGE-INFANT-MONTHS",
        new_vitals={"temperature": 100.4},
        complaint_update="Fever",
    )

    assert updated["tier"] == 2
    assert "RF_HIGH_FEVER_INFANT" in updated["red_flags"]
    db.clear_all()


def test_smart_ticket_escapes_patient_content_and_does_not_invent_wait_time():
    ticket = render_smart_token_ticket(
        {
            "encounter_id": "ENC-TEST-1",
            "patient_name": '<script>alert("x")</script>',
            "department": '<img src=x onerror="alert(1)">',
            "created_at": "2026-10-28T10:00:00",
            "tier": 2,
            "abstained": False,
            "vitals": {},
        },
        show_staff_clinical_info=True,
    )

    assert "<script>" not in ticket
    assert "<img src=x" not in ticket
    assert "&lt;script&gt;" in ticket
    assert "Wait estimate unavailable" in ticket
    assert "Est. Wait: ~" not in ticket


def test_official_organizer_data_and_evaluator_are_detected_separately_from_cache():
    status = evaluate_runner.check_organizer_data_status()

    if not status["has_evaluate_script"] or not status["has_organizer_data"]:
        pytest.skip("Organizer evaluator and datasets are excluded from the reduced package.")

    assert status["has_evaluate_script"] is True
    assert status["has_organizer_data"] is True
    assert status["is_placeholder"] is False
    assert status["organizer_data_file"].endswith("intake_train.csv")
    assert Path(status["evaluate_script_path"]).parts[-3:] == ("data", "organizer", "evaluate.py")
    assert evaluate_runner.execute_organizer_evaluate_script() is None
