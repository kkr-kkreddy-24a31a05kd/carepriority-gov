from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import hackathon_pipeline
from hackathon_pipeline import clean_vitals_df, feature_frame, safety_predictions

BASELINE_PATH = (
    Path(hackathon_pipeline.ORGANIZER_ROOT) / "baseline" / "baseline.py"
)


@pytest.fixture(autouse=True)
def require_organizer_baseline(request):
    if (
        request.node.name not in {
            "test_pipeline_import_is_safe_without_organizer_baseline",
            "test_missing_organizer_baseline_reports_required_file",
        }
        and not BASELINE_PATH.is_file()
    ):
        pytest.skip(
            "Organizer-pipeline behavior tests require the separately supplied "
            f"organizer baseline: {BASELINE_PATH}"
        )


def test_pipeline_import_is_safe_without_organizer_baseline():
    assert callable(hackathon_pipeline.main)


def test_missing_organizer_baseline_reports_required_file():
    if BASELINE_PATH.is_file():
        pytest.skip("Organizer baseline is present; missing-file behavior is not applicable.")

    with pytest.raises(FileNotFoundError, match="Organizer baseline is missing"):
        clean_vitals_df(intake_row())


class FixedModel:
    def __init__(self, classes, probabilities):
        self.classes_ = np.asarray(classes)
        self.probabilities = np.asarray(probabilities)

    def predict_proba(self, features):
        return np.tile(self.probabilities, (len(features), 1))


def intake_row(**overrides):
    row = {
        "visit_id": "OPD-TEST-1",
        "arrival_ts": "2026-08-03T08:00:00",
        "age_years": 30,
        "age_months": np.nan,
        "sex": "M",
        "pregnant": "",
        "arrival_mode": "walk-in",
        "bp_systolic": 120,
        "bp_diastolic": 80,
        "pulse_bpm": 75,
        "temp_c": 36.8,
        "spo2_pct": 98,
        "chief_complaint": "Routine outpatient visit",
        "known_conditions": "",
        "current_medications": "",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def models():
    return (
        FixedModel([1, 2, 3, 4, 5], [0.01, 0.02, 0.9, 0.04, 0.03]),
        FixedModel(["General Medicine"], [1.0]),
    )


def test_critical_vital_escalation_precedes_missing_complaint_abstention():
    triage_model, department_model = models()
    result = safety_predictions(
        intake_row(chief_complaint="", spo2_pct=80),
        triage_model,
        department_model,
    ).iloc[0]

    assert result["triage_level"] == 1
    assert result["department"] == "Emergency/Casualty"


def test_incomplete_non_emergency_intake_is_sent_to_nurse():
    triage_model, department_model = models()
    result = safety_predictions(
        intake_row(
            chief_complaint="",
            age_years=np.nan,
            bp_systolic=0,
            bp_diastolic=0,
            pulse_bpm=np.nan,
            temp_c=np.nan,
            spo2_pct=np.nan,
        ),
        triage_model,
        department_model,
    ).iloc[0]

    assert result["triage_level"] == "ABSTAIN"
    assert result["department"] == ""


def test_empty_complaint_with_multiple_usable_vitals_is_not_automatically_abstained():
    triage_model, department_model = models()
    result = safety_predictions(
        intake_row(chief_complaint=""),
        triage_model,
        department_model,
    ).iloc[0]

    assert result["triage_level"] == 3
    assert result["department"] == "General Medicine"


def test_calibrated_emergency_threshold_can_only_escalate_model_output():
    triage_model, department_model = models()
    result = safety_predictions(
        intake_row(),
        triage_model,
        department_model,
        emergency_threshold=0.02,
    ).iloc[0]

    assert result["triage_level"] == 2
    assert result["department"] == "Emergency/Casualty"


def test_critically_low_recorded_systolic_pressure_escalates_before_cleaning():
    triage_model, department_model = models()
    result = safety_predictions(
        intake_row(bp_systolic=40, bp_diastolic=20),
        triage_model,
        department_model,
    ).iloc[0]

    assert result["triage_level"] == 1
    assert result["department"] == "Emergency/Casualty"


def test_self_harm_red_flag_uses_emergency_department():
    triage_model, department_model = models()
    result = safety_predictions(
        intake_row(chief_complaint="cut my wrist and do not want to live"),
        triage_model,
        department_model,
    ).iloc[0]

    assert result["triage_level"] == 2
    assert result["department"] == "Emergency/Casualty"
    assert "RF_SUICIDAL" in result["red_flags"]


@pytest.mark.parametrize(
    ("complaint", "conditions", "age", "expected_flag"),
    [
        (
            "mujhe 6 din se taap hai aur baar baar ulti, kuch nahi tik raha",
            "",
            30,
            "RF_DENGUE_WARNING",
        ),
        (
            "en akkavukku 4 naala kaaichal irukku, vayiru vali, thalai suthudhu",
            "",
            40,
            "RF_DENGUE_WARNING",
        ),
        (
            "achanak ek aankh ki roshni chali gayi",
            "",
            26,
            "RF_EYE_EMERGENCY",
        ),
        (
            "अचानक मुझे हाथ में जान नहीं है",
            "",
            80,
            "RF_STROKE_SIGNS",
        ),
        (
            "katori bhar khansi mein khoon aaya",
            "",
            84,
            "RF_SEVERE_BLEEDING",
        ),
        (
            "heavy feeling in centre of chest and profuse sweating",
            "",
            55,
            "RF_CHEST_PAIN_CARDIAC",
        ),
        (
            "restless uneasy feeling and burning in stomach",
            "sugar ki bimari; hypertension",
            71,
            "RF_CHEST_PAIN_CARDIAC",
        ),
        (
            "neerasam ga undi, chematalu padutunnayi, aayasam, kallu tirugutunnayi",
            "DM",
            68,
            "RF_CHEST_PAIN_CARDIAC",
        ),
        (
            "कमजोरी है और ठंडा पसीना आ रहा है और चक्कर आ रहे हैं",
            "madhumeh",
            66,
            "RF_CHEST_PAIN_CARDIAC",
        ),
        (
            "khet mein keetnasshak chhidakte samay jee machal raha hai, ulti",
            "",
            42,
            "RF_POISONING",
        ),
        (
            "2 ghonta aage onek pure gecheh",
            "",
            26,
            "RF_BURNS_MAJOR",
        ),
    ],
)
def test_primer_red_flags_cover_documented_multilingual_presentations(
    complaint, conditions, age, expected_flag
):
    triage_model, department_model = models()
    result = safety_predictions(
        intake_row(
            chief_complaint=complaint,
            known_conditions=conditions,
            age_years=age,
        ),
        triage_model,
        department_model,
    ).iloc[0]

    assert result["triage_level"] == 2
    assert result["department"] == "Emergency/Casualty"
    assert expected_flag in result["red_flags"]


def test_routine_gas_complaint_in_young_adult_is_not_cardiac_red_flag():
    triage_model, department_model = models()
    result = safety_predictions(
        intake_row(chief_complaint="gas and burning in stomach", age_years=30),
        triage_model,
        department_model,
    ).iloc[0]

    assert result["triage_level"] == 3
    assert "RF_CHEST_PAIN_CARDIAC" not in result["red_flags"]


def test_noncardiac_chest_pain_and_minor_sputum_blood_are_not_emergency_flags():
    triage_model, department_model = models()
    cough_related_pain = safety_predictions(
        intake_row(chief_complaint="chest pain when coughing"),
        triage_model,
        department_model,
    ).iloc[0]
    minor_sputum_blood = safety_predictions(
        intake_row(chief_complaint="chronic cough with blood streaks in sputum"),
        triage_model,
        department_model,
    ).iloc[0]
    fever_with_diarrhea = safety_predictions(
        intake_row(chief_complaint="mild fever, loose motions and vomiting"),
        triage_model,
        department_model,
    ).iloc[0]

    assert cough_related_pain["triage_level"] == 3
    assert "RF_CHEST_PAIN_CARDIAC" not in cough_related_pain["red_flags"]
    assert minor_sputum_blood["triage_level"] == 3
    assert "RF_SEVERE_BLEEDING" not in minor_sputum_blood["red_flags"]
    assert fever_with_diarrhea["triage_level"] == 3
    assert "RF_DENGUE_WARNING" not in fever_with_diarrhea["red_flags"]


def test_stopped_nosebleed_does_not_match_uncontrolled_bleeding_rule():
    triage_model, department_model = models()
    stopped = safety_predictions(
        intake_row(chief_complaint="naak se khoon aaya tha, ab band hai"),
        triage_model,
        department_model,
    ).iloc[0]
    uncontrolled = safety_predictions(
        intake_row(chief_complaint="khoon band nahi ho raha"),
        triage_model,
        department_model,
    ).iloc[0]

    assert stopped["triage_level"] == 3
    assert "RF_SEVERE_BLEEDING" not in stopped["red_flags"]
    assert uncontrolled["triage_level"] == 1
    assert uncontrolled["department"] == "Emergency/Casualty"


def test_documented_fahrenheit_entry_in_celsius_column_is_normalized():
    cleaned = clean_vitals_df(intake_row(temp_c=96.0))
    features = feature_frame(intake_row(temp_c=96.0))

    expected_celsius = (96.0 - 32.0) * 5.0 / 9.0
    assert cleaned.loc[0, "temp"] == expected_celsius
    assert features.loc[0, "temp_c"] == expected_celsius


def test_official_organizer_temperature_and_pediatric_vital_rules():
    triage_model, department_model = models()
    infant_hypothermia = safety_predictions(
        intake_row(age_years=0, age_months=2, temp_c=35.5),
        triage_model,
        department_model,
    ).iloc[0]
    child_bradycardia = safety_predictions(
        intake_row(age_years=7, age_months=np.nan, pulse_bpm=55),
        triage_model,
        department_model,
    ).iloc[0]

    assert infant_hypothermia["triage_level"] == 2
    assert infant_hypothermia["department"] == "Emergency/Casualty"
    assert child_bradycardia["triage_level"] == 1
    assert child_bradycardia["department"] == "Emergency/Casualty"


def test_invalid_vital_values_are_removed_from_model_features():
    features = feature_frame(
        intake_row(bp_systolic=0, bp_diastolic=0, pulse_bpm=1544, temp_c=391, spo2_pct=999)
    )

    assert pd.isna(features.loc[0, "bp_systolic"])
    assert pd.isna(features.loc[0, "pulse_bpm"])
    assert pd.isna(features.loc[0, "temp_c"])
    assert pd.isna(features.loc[0, "spo2_pct"])
