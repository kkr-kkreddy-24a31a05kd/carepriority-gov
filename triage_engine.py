"""CarePriority Gov - Clinical Triage & Guardrail Engine
======================================================
Pure functions, zero UI dependencies.
Architecture:
1. Data-entry validation & unit-error detection for vitals
   - BP of 0 = treated as missing, flagged for verification, stored as None.
   - Temperature: if unit is Celsius and value > 43 (e.g. 96 C), flagged as probable unit error (likely Fahrenheit).
   - Missing vitals stay None, never coerced to 0.
2. Model First: Scikit-learn TF-IDF char n-grams + numeric vitals pipeline
   - Asymmetric 20x safety thresholding prioritizing emergency sensitivity/recall.
3. Hard-Rule Layer on Top:
   - Mandatory physiological triggers and multilingual red-flag keywords (English, Hindi, Telugu, Hinglish, Telugu-Latin).
   - Can ONLY ESCALATE, never downgrade.
4. Safety Abstention Logic:
   - Missing data, ambiguous complaints, or low confidence.
   - NEVER defaults to Level 5 Non-Urgent; routed to Needs Nurse Review.
5. Department Recommendation & Explainability Tree.
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

from rules_config import (
    TRIAGE_LEVELS,
    VITALS_PLAUSIBILITY,
    HARD_ESCALATION_VITALS,
    RED_FLAG_KEYWORDS,
    SECONDARY_KEYWORDS,
    DEPARTMENTS,
    DEPARTMENT_RULES,
    ABSTENTION_CONDITIONS,
    ASYMMETRIC_COST_CONFIG
)

# Global model cache to avoid repeated disk I/O
_CACHED_TRIAGE_PIPELINE = None
_CACHED_DEPT_PIPELINE = None
_CACHED_METADATA = None
_TRIAGE_MODEL_LOAD_ATTEMPTED = False


def load_ml_models() -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, Any]]]:
    """Loads saved scikit-learn models from models/ directory if available."""
    global _CACHED_TRIAGE_PIPELINE, _CACHED_DEPT_PIPELINE, _CACHED_METADATA, _TRIAGE_MODEL_LOAD_ATTEMPTED

    if _TRIAGE_MODEL_LOAD_ATTEMPTED:
        return _CACHED_TRIAGE_PIPELINE, _CACHED_DEPT_PIPELINE, _CACHED_METADATA

    _TRIAGE_MODEL_LOAD_ATTEMPTED = True
    triage_path = "models/triage_pipeline.joblib"
    dept_path = "models/dept_pipeline.joblib"
    meta_path = "models/model_metadata.joblib"

    if os.path.exists(triage_path) and os.path.exists(dept_path):
        try:
            _CACHED_TRIAGE_PIPELINE = joblib.load(triage_path)
            _CACHED_DEPT_PIPELINE = joblib.load(dept_path)
            if os.path.exists(meta_path):
                _CACHED_METADATA = joblib.load(meta_path)
        except Exception as e:
            print(f"[CarePriority Engine] Note: Could not load ML models: {e}")

    return _CACHED_TRIAGE_PIPELINE, _CACHED_DEPT_PIPELINE, _CACHED_METADATA


def check_model_status() -> Dict[str, Any]:
    """
    Checks whether trained scikit-learn models are present in models/ directory.
    Returns status and informative banner text. Never modifies triage logic.
    """
    triage_path = "models/triage_pipeline.joblib"
    dept_path = "models/dept_pipeline.joblib"
    meta_path = "models/model_metadata.joblib"
    has_triage = os.path.exists(triage_path)
    has_dept = os.path.exists(dept_path)
    has_meta = os.path.exists(meta_path)
    has_models = has_triage and has_dept
    return {
        "has_models": has_models,
        "has_triage": has_triage,
        "has_dept": has_dept,
        "has_meta": has_meta,
        "status_msg": "ML models available (models/ loaded)" if has_models else "Trained model files not found (heuristic clinical rules active)"
    }


def clean_vitals_for_storage(
    bp_systolic: Optional[int],
    bp_diastolic: Optional[int],
    spo2: Optional[float],
    pulse: Optional[int],
    temperature: Optional[float]
) -> Dict[str, Optional[float]]:
    """
    Cleans vitals according to strict data entry rules:
    - BP of 0 = treated as missing, stored as None.
    - Missing values stay None, never coerced to 0.
    """
    sbp = None if (bp_systolic is None or bp_systolic == 0) else int(bp_systolic)
    dbp = None if (bp_diastolic is None or bp_diastolic == 0 or bp_systolic == 0) else int(bp_diastolic)
    return {
        "bp_systolic": sbp,
        "bp_diastolic": dbp,
        "spo2": spo2,
        "pulse": pulse,
        "temperature": temperature
    }


def validate_vitals(
    age: Optional[int],
    spo2: Optional[float],
    bp_systolic: Optional[int],
    bp_diastolic: Optional[int],
    pulse: Optional[int],
    temperature: Optional[float],
    temp_unit: str = "F"
) -> Tuple[bool, List[str], List[str]]:
    """
    Validates entered vitals for clinical plausibility and entry errors.
    Returns: (is_valid, error_messages, warning_messages)

    DATA-ENTRY RULES:
    - BP of 0 = treated as missing, flagged for verification, stored as None.
    - Temperature: if unit is Celsius and value is above 43 (e.g. 96 C), flag as
      probable unit error (likely Fahrenheit). Never silently trust it. If unit is
      Fahrenheit and value is outside a plausible human range (90-110F), also flag it.
    - Missing values stay None, never coerced to 0.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Age validation
    if age is not None:
        if age < VITALS_PLAUSIBILITY["age"]["min"] or age > VITALS_PLAUSIBILITY["age"]["max"]:
            errors.append(VITALS_PLAUSIBILITY["age"]["error_msg"])

    # 2. SpO2 validation
    if spo2 is not None:
        if spo2 < VITALS_PLAUSIBILITY["spo2"]["min"] or spo2 > VITALS_PLAUSIBILITY["spo2"]["max"]:
            errors.append(VITALS_PLAUSIBILITY["spo2"]["error_msg"])

    # 3. Blood Pressure validation
    if bp_systolic is not None:
        if bp_systolic == 0:
            warnings.append(VITALS_PLAUSIBILITY["bp_systolic"]["zero_warning"])
        elif bp_systolic < VITALS_PLAUSIBILITY["bp_systolic"]["min"] or bp_systolic > VITALS_PLAUSIBILITY["bp_systolic"]["max"]:
            errors.append(f"Systolic BP ({bp_systolic} mmHg) is outside plausible range (40–300 mmHg).")

    if bp_diastolic is not None:
        if bp_diastolic == 0:
            warnings.append("Diastolic BP of 0 detected: Treated as missing/unobtainable and stored as None.")
        elif bp_diastolic < VITALS_PLAUSIBILITY["bp_diastolic"]["min"] or bp_diastolic > VITALS_PLAUSIBILITY["bp_diastolic"]["max"]:
            errors.append(f"Diastolic BP ({bp_diastolic} mmHg) is outside plausible range (20–200 mmHg).")

    if bp_systolic is not None and bp_diastolic is not None and bp_systolic > 0 and bp_diastolic > 0:
        if bp_diastolic >= bp_systolic:
            errors.append(f"Diastolic BP ({bp_diastolic}) cannot be greater than or equal to Systolic BP ({bp_systolic}).")

    # 4. Pulse validation
    if pulse is not None:
        if pulse < VITALS_PLAUSIBILITY["pulse"]["min"] or pulse > VITALS_PLAUSIBILITY["pulse"]["max"]:
            errors.append(f"Pulse rate ({pulse} bpm) is outside plausible range (20–300 bpm).")

    # 5. Temperature plausibility & unit error check
    if temperature is not None:
        unit = temp_unit.upper() if temp_unit else "F"

        # Explicit Celsius input check
        if unit == "C":
            if temperature > VITALS_PLAUSIBILITY["temperature"]["celsius_max_plausible"]:
                errors.append(
                    f"Temperature of {temperature}°C entered in Celsius is above 43°C (e.g. 96°C) — "
                    "flagged as a probable unit error (likely intended as Fahrenheit). Never silently trusted."
                )
            elif temperature < 30.0:
                errors.append(f"Temperature of {temperature}°C is below plausible human limit in Celsius.")
        else:
            # Fahrenheit input checks
            min_f = VITALS_PLAUSIBILITY["temperature"]["min_fahrenheit"]
            max_f = VITALS_PLAUSIBILITY["temperature"]["max_fahrenheit"]
            c_low = VITALS_PLAUSIBILITY["temperature"]["probable_celsius_low"]
            c_high = VITALS_PLAUSIBILITY["temperature"]["probable_celsius_high"]

            # If user enters 34 to 43 without declaring Celsius, flag as probable Celsius error
            if c_low <= temperature <= c_high:
                errors.append(
                    f"Temperature of {temperature}° is in Celsius range (34°C–43°C). "
                    f"Please convert to Fahrenheit (e.g. {temperature * 9/5 + 32:.1f}°F)."
                )
            elif temperature < min_f or temperature > max_f:
                errors.append(
                    f"Temperature ({temperature}°F) is outside plausible human range ({min_f}°F–{max_f}°F). "
                    "Flagged as a probable unit/entry error rather than accepted."
                )

    return (len(errors) == 0, errors, warnings)


def detect_symptom_matches(text: str, keyword_list: List[str]) -> List[str]:
    """
    Matches keywords in free text preserving Unicode.
    Covers English, Hindi, Telugu, Hinglish, and Telugu-in-Latin-script spellings.
    Case-insensitive for Latin strings; exact substring match for Unicode scripts.
    """
    if not text:
        return []

    text_lower = text.lower()
    matches = []
    for kw in keyword_list:
        kw_clean = kw.strip()
        if not kw_clean:
            continue
        if kw_clean.lower() in text_lower:
            matches.append(kw_clean)

    return list(dict.fromkeys(matches))


def evaluate_safety_abstention(
    complaint: Optional[str],
    spo2: Optional[float],
    bp_systolic: Optional[int],
    pulse: Optional[int],
    temperature: Optional[float],
    model_conf: Optional[float] = None
) -> Tuple[bool, Optional[str]]:
    """
    Safety Abstention Guardrail:
    If data is incomplete (no complaint AND missing key vitals), ambiguous,
    contradictory, or model confidence is low with missing clinical indicators:
    the system MUST abstain and flag for immediate nurse review.
    Abstention must NEVER default to Level 5.
    """
    complaint_clean = (complaint or "").strip()
    has_complaint = len(complaint_clean) > 2
    has_any_vital = any(v is not None for v in [spo2, bp_systolic, pulse, temperature])
    has_key_vitals = (spo2 is not None) or (pulse is not None) or (bp_systolic is not None)

    # Condition 1: Missing both complaint and all vitals
    if not has_complaint and not has_any_vital:
        return True, "No complaint provided and no vital signs entered. Incomplete data requires in-person nurse assessment."

    # Condition 2: No complaint and missing all key triage vitals
    if not has_complaint and not has_key_vitals:
        return True, "Missing chief complaint and all cardiovascular/respiratory vitals. Send to triage nurse."

    # Condition 3: Very short/unclear non-clinical token with no vitals
    if has_complaint and len(complaint_clean) < 4 and not has_any_vital:
        return True, "Ambiguous intake with insufficient complaint detail and zero recorded vitals."

    # Condition 4: Contradiction: BP 0 but patient has pulse > 50 without emergency tag
    if bp_systolic == 0 and pulse is not None and pulse > 50:
        return True, "Contradictory hemodynamic record: Systolic BP recorded as 0 mmHg with active pulse. Requires immediate manual verification."

    # Condition 5: Low model confidence (< 0.35) combined with missing key vitals
    if model_conf is not None and model_conf < ABSTENTION_CONDITIONS["low_model_confidence"]["threshold"] and not has_key_vitals:
        return True, "Low triage model confidence with missing primary vital signs. Send to triage nurse for manual assessment."

    return False, None


def evaluate_hard_escalations(
    spo2: Optional[float],
    bp_systolic: Optional[int],
    bp_diastolic: Optional[int],
    pulse: Optional[int],
    temperature: Optional[float],
    complaint: Optional[str]
) -> Tuple[Optional[int], Optional[str], List[str], List[str]]:
    """
    MANDATORY HARD-CODED ESCALATION
    Cannot be overridden downward by any model or scoring rule.
    Returns: (forced_tier, rule_name, reasons, red_flags_detected)
    Forces Level 1 or Level 2 immediately when critical signs or red-flags are found.
    """
    reasons: List[str] = []
    red_flags: List[str] = []
    forced_tier: Optional[int] = None
    rule_name: Optional[str] = None

    complaint_str = (complaint or "")

    # 1. Level 1 Hard Symptom Keywords (Cardiac arrest, Unconscious, Not breathing, etc.)
    l1_matches = detect_symptom_matches(complaint_str, RED_FLAG_KEYWORDS["L1_SYMPTOMS"]["keywords"])
    if l1_matches:
        forced_tier = 1
        rule_name = RED_FLAG_KEYWORDS["L1_SYMPTOMS"]["rule_name"]
        red_flags.extend(l1_matches)
        reasons.append(f"ESCALATION RULE TRIGGERED: Critical resuscitation keyword detected ({', '.join(l1_matches)})")

    # 2. Level 1 Physiological Vitals
    l1_vitals = HARD_ESCALATION_VITALS["L1"]
    if spo2 is not None and spo2 < l1_vitals["spo2_critical"]:
        forced_tier = 1
        rule_name = "Critical Anoxia (SpO2 < 85%)"
        reasons.append(f"ESCALATION RULE TRIGGERED: Profound hypoxemia (SpO2 {spo2}% < 85%)")

    if pulse is not None:
        if pulse > l1_vitals["pulse_arrest_high"]:
            forced_tier = 1
            rule_name = f"Extreme Tachyarrhythmia (Pulse {pulse} bpm)"
            reasons.append(f"ESCALATION RULE TRIGGERED: Extreme pulse rate ({pulse} bpm > 160 bpm)")
        elif pulse < l1_vitals["pulse_arrest_low"]:
            forced_tier = 1
            rule_name = f"Critical Bradycardia / Near-Arrest (Pulse {pulse} bpm)"
            reasons.append(f"ESCALATION RULE TRIGGERED: Extreme bradycardia ({pulse} bpm < 35 bpm)")

    if bp_systolic is not None and 0 < bp_systolic < l1_vitals["bp_systolic_shock"]:
        forced_tier = 1
        rule_name = f"Profound Decompensated Shock (SBP {bp_systolic} mmHg)"
        reasons.append(f"ESCALATION RULE TRIGGERED: Severe hypotension / profound shock (SBP {bp_systolic} < 70 mmHg)")

    if temperature is not None and temperature >= l1_vitals["temp_hyperpyrexia"]:
        forced_tier = 1
        rule_name = f"Hyperpyrexia Crisis (Temp {temperature}°F)"
        reasons.append(f"ESCALATION RULE TRIGGERED: Severe hyperpyrexia ({temperature}°F >= 106.0°F)")

    # If already forced to Level 1, return immediately (cannot escalate higher)
    if forced_tier == 1:
        return 1, rule_name or "Level 1 Resuscitation", reasons, red_flags

    # 3. Level 2 Hard Symptom Keywords (Chest pain, stroke, poisoning, snake bite, etc.)
    l2_matches = detect_symptom_matches(complaint_str, RED_FLAG_KEYWORDS["L2_SYMPTOMS"]["keywords"])
    if l2_matches:
        forced_tier = 2
        rule_name = RED_FLAG_KEYWORDS["L2_SYMPTOMS"]["rule_name"]
        red_flags.extend(l2_matches)
        reasons.append(f"ESCALATION RULE TRIGGERED: Emergent red-flag keyword ({', '.join(l2_matches)})")

    # 4. Level 2 Physiological Vitals
    l2_vitals = HARD_ESCALATION_VITALS["L2"]

    # SpO2 < 92% MANDATORY L2 ESCALATION
    if spo2 is not None and spo2 < l2_vitals["spo2_hypoxia"]:
        forced_tier = 2
        rule_name = f"Mandatory Hypoxia Escalation (SpO2 {spo2}% < 92%)"
        reasons.append(f"ESCALATION RULE TRIGGERED: Hypoxemia (SpO2 {spo2}% < 92%)")

    # Severe Hypotension SBP < 90
    if bp_systolic is not None and 0 < bp_systolic < l2_vitals["bp_systolic_hypotension"]:
        forced_tier = 2
        rule_name = f"Hemodynamic Compromise (SBP {bp_systolic} mmHg)"
        reasons.append(f"ESCALATION RULE TRIGGERED: Severe hypotension (SBP {bp_systolic} mmHg < 90 mmHg)")

    # Hypertensive crisis SBP >= 180 or DBP >= 120
    if bp_systolic is not None and bp_systolic >= l2_vitals["bp_systolic_crisis"]:
        forced_tier = 2
        rule_name = f"Hypertensive Urgency/Crisis (SBP {bp_systolic} mmHg)"
        reasons.append(f"ESCALATION RULE TRIGGERED: Hypertensive crisis (SBP {bp_systolic} mmHg >= 180)")

    if bp_diastolic is not None and bp_diastolic >= l2_vitals["bp_diastolic_crisis"]:
        forced_tier = 2
        rule_name = f"Hypertensive Crisis (DBP {bp_diastolic} mmHg)"
        reasons.append(f"ESCALATION RULE TRIGGERED: Hypertensive crisis (DBP {bp_diastolic} mmHg >= 120)")

    # Pulse extremes
    if pulse is not None:
        if pulse >= l2_vitals["pulse_tachycardia"]:
            forced_tier = 2
            rule_name = f"Severe Tachycardia (Pulse {pulse} bpm)"
            reasons.append(f"ESCALATION RULE TRIGGERED: Severe tachycardia ({pulse} bpm >= 130 bpm)")
        elif pulse <= l2_vitals["pulse_bradycardia"]:
            forced_tier = 2
            rule_name = f"Marked Bradycardia (Pulse {pulse} bpm)"
            reasons.append(f"ESCALATION RULE TRIGGERED: Marked bradycardia ({pulse} bpm <= 45 bpm)")

    # Temperature extremes
    if temperature is not None:
        if temperature >= l2_vitals["temp_high_fever"]:
            forced_tier = 2
            rule_name = f"High Grade Pyrexia (Temp {temperature}°F)"
            reasons.append(f"ESCALATION RULE TRIGGERED: High fever ({temperature}°F >= 103.5°F)")
        elif temperature < l2_vitals["temp_hypothermia"]:
            forced_tier = 2
            rule_name = f"Hypothermia (Temp {temperature}°F)"
            reasons.append(f"ESCALATION RULE TRIGGERED: Hypothermia ({temperature}°F < 94.0°F)")

    if forced_tier == 2:
        return 2, rule_name or "Level 2 Emergent Escalation", reasons, red_flags

    return None, None, reasons, red_flags


def score_tier_without_hard_escalation(
    spo2: Optional[float],
    bp_systolic: Optional[int],
    pulse: Optional[int],
    temperature: Optional[float],
    complaint: Optional[str]
) -> Tuple[int, List[str]]:
    """
    Transparent clinical scoring heuristics for Levels 3, 4, 5.
    Used as baseline fallback when ML model is not available or for comparative evaluation.
    """
    reasons: List[str] = []
    complaint_str = (complaint or "").lower()

    has_fever = (temperature is not None and 100.4 <= temperature < 103.5)
    has_mild_tachy = (pulse is not None and 100 <= pulse < 130)
    has_mild_hypo = (bp_systolic is not None and 90 <= bp_systolic < 100)
    has_mild_spo2_dip = (spo2 is not None and 92 <= spo2 <= 94)

    l3_matches = detect_symptom_matches(complaint_str, SECONDARY_KEYWORDS["L3_URGENT"])
    l4_matches = detect_symptom_matches(complaint_str, SECONDARY_KEYWORDS["L4_LESS_URGENT"])
    l5_matches = detect_symptom_matches(complaint_str, SECONDARY_KEYWORDS["L5_NON_URGENT"])

    if l3_matches:
        reasons.append(f"Urgent symptom keyword matched: {', '.join(l3_matches)}")
        return 3, reasons

    if sum([bool(has_fever), bool(has_mild_tachy), bool(has_mild_hypo), bool(has_mild_spo2_dip)]) >= 2:
        vital_details = []
        if has_fever: vital_details.append(f"Fever {temperature}°F")
        if has_mild_tachy: vital_details.append(f"Pulse {pulse} bpm")
        if has_mild_hypo: vital_details.append(f"SBP {bp_systolic} mmHg")
        if has_mild_spo2_dip: vital_details.append(f"Borderline SpO2 {spo2}%")
        reasons.append(f"Multiple borderline/mild vital abnormalities: {', '.join(vital_details)}")
        return 3, reasons

    if has_fever:
        reasons.append(f"Significant pyrexia ({temperature}°F) requiring acute medical evaluation")
        return 3, reasons

    if has_mild_spo2_dip:
        reasons.append(f"Borderline SpO2 ({spo2}%) requiring monitoring")
        return 3, reasons

    if l4_matches:
        reasons.append(f"Less urgent clinical complaint matched: {', '.join(l4_matches)}")
        return 4, reasons

    if has_mild_tachy:
        reasons.append(f"Mild tachycardia ({pulse} bpm) with stable other vitals")
        return 4, reasons

    if l5_matches:
        reasons.append(f"Routine / non-urgent checkup keyword matched: {', '.join(l5_matches)}")
        return 5, reasons

    reasons.append("All measured vitals within normal physiological limits; mild/routine presentation")
    return 4, reasons


def recommend_department(
    complaint: Optional[str],
    age: Optional[int],
    gender: Optional[str],
    tier: Optional[int]
) -> str:
    """Recommends clinical department based on transparent rule priority across 13 canonical departments."""
    if tier == 1:
        return "Emergency/Casualty"

    complaint_str = (complaint or "").lower()

    # 1. Specialized physical/organ injuries checked first (Dental, Ophthalmology, ENT, Orthopaedics)
    organ_specialties = ["Dental", "Ophthalmology", "ENT", "Orthopaedics"]
    for spec in organ_specialties:
        rule = DEPARTMENT_RULES.get(spec)
        if rule and "keywords" in rule:
            matches = detect_symptom_matches(complaint_str, rule["keywords"])
            if matches:
                return spec

    # 2. Paediatrics age threshold (children under 14 with medical/fever/cough/respiratory complaints)
    paed_rule = DEPARTMENT_RULES.get("Paediatrics") or DEPARTMENT_RULES.get("Pediatrics", {})
    if age is not None and age <= paed_rule.get("max_age", 14):
        return "Paediatrics"

    # 3. Adult specialties: Cardiology, Pulmonology, General Surgery, Dermatology, Psychiatry
    adult_specialties = [
        "Cardiology", "Pulmonology/Chest & TB", "General Surgery",
        "Dermatology", "Psychiatry"
    ]
    for spec in adult_specialties:
        rule = DEPARTMENT_RULES.get(spec)
        if rule and "keywords" in rule:
            matches = detect_symptom_matches(complaint_str, rule["keywords"])
            if matches:
                return spec

    # 4. Obstetrics & Gynaecology (for female patients)
    if gender and gender.lower() in ["female", "f"]:
        obgyn_rule = DEPARTMENT_RULES.get("Obstetrics & Gynaecology") or DEPARTMENT_RULES.get("Obstetrics & Gynecology", {})
        if "keywords" in obgyn_rule:
            obgyn_matches = detect_symptom_matches(complaint_str, obgyn_rule["keywords"])
            if obgyn_matches:
                return "Obstetrics & Gynaecology"

    return "General Medicine"


def predict_ml_triage(
    complaint: str,
    age: Optional[int],
    gender: Optional[str],
    spo2: Optional[float],
    bp_systolic: Optional[int],
    bp_diastolic: Optional[int],
    pulse: Optional[int],
    temperature: Optional[float],
    known_conditions: Optional[str] = None,
    medications: Optional[str] = None
) -> Tuple[Optional[int], Optional[float], Optional[np.ndarray], Optional[str]]:
    """
    Executes trained scikit-learn models using optimal asymmetric emergency threshold.
    Returns: (predicted_tier, confidence, probabilities, predicted_dept)
    """
    triage_pipe, dept_pipe, metadata = load_ml_models()
    if triage_pipe is None:
        return None, None, None, None

    sample_df = pd.DataFrame([{
        "complaint": complaint or "",
        "known_conditions": known_conditions or "",
        "medications": medications or "",
        "spo2": spo2,
        "bp_systolic": bp_systolic,
        "bp_diastolic": bp_diastolic,
        "pulse": pulse,
        "temperature": temperature,
        "age": age,
        "sex": gender or "Unknown"
    }])

    try:
        probs = triage_pipe.predict_proba(sample_df)[0]
        # In classes_ [1, 2, 3, 4, 5], idx 0 is L1, idx 1 is L2
        p_emerg = probs[0] + probs[1]
        opt_tau = metadata.get("optimal_emergency_threshold", 0.50) if metadata else 0.50

        # Apply asymmetric decision threshold
        if p_emerg >= opt_tau:
            pred_tier = 1 if probs[0] > probs[1] else 2
        else:
            # Predict most probable among L3, L4, L5
            sub_probs = probs[2:]
            pred_tier = int(3 + np.argmax(sub_probs))

        confidence = float(np.max(probs))

        pred_dept = None
        if dept_pipe is not None:
            pred_dept = str(dept_pipe.predict(sample_df)[0])

        return pred_tier, confidence, probs, pred_dept
    except Exception as e:
        print(f"[CarePriority Engine] ML inference error: {e}")
        return None, None, None, None


def run_triage(
    patient_id: str,
    patient_name: Optional[str],
    age: Optional[int],
    gender: Optional[str],
    spo2: Optional[float],
    bp_systolic: Optional[int],
    bp_diastolic: Optional[int],
    pulse: Optional[int],
    temperature: Optional[float],
    complaint: Optional[str],
    known_conditions: Optional[str] = None,
    medications: Optional[str] = None,
    temp_unit: str = "F"
) -> Dict[str, Any]:
    """
    Complete hybrid clinical triage pipeline:
    1. Validation & Data-Entry checks (BP 0, Celsius errors, plausible ranges)
    2. Model First: Scikit-learn triage & department pipeline with 20x asymmetric threshold
    3. Hard-Rule Layer on Top: MANDATORY escalation (can only escalate, never downgrade)
    4. Safety Abstention Logic: Missing/contradictory/low-confidence data routes to nurse
    5. Explainability tree & red-flag chips.
    """
    # 1. Validation
    is_valid, validation_errors, validation_warnings = validate_vitals(
        age=age,
        spo2=spo2,
        bp_systolic=bp_systolic,
        bp_diastolic=bp_diastolic,
        pulse=pulse,
        temperature=temperature,
        temp_unit=temp_unit
    )

    if not is_valid:
        return {
            "encounter_id": patient_id,
            "tier": None,
            "tier_name": "Needs Nurse Review",
            "department": "Triage Review Desk",
            "abstained": True,
            "abstention_reason": "Vitals validation failed: " + "; ".join(validation_errors),
            "reasons": validation_errors,
            "warnings": validation_warnings,
            "red_flags": [],
            "is_escalated": False,
            "escalation_rule": None,
            "model_tier": None,
            "model_confidence": None,
            "cleaned_vitals": clean_vitals_for_storage(bp_systolic, bp_diastolic, spo2, pulse, temperature)
        }

    # Clean vitals (BP of 0 treated as missing and stored as None)
    cleaned_vitals = clean_vitals_for_storage(bp_systolic, bp_diastolic, spo2, pulse, temperature)
    effective_sbp = cleaned_vitals["bp_systolic"]
    effective_dbp = cleaned_vitals["bp_diastolic"]

    # 2. MODEL FIRST: Predict triage tier and department with ML pipeline
    model_tier, model_conf, model_probs, model_dept = predict_ml_triage(
        complaint=complaint or "",
        age=age,
        gender=gender,
        spo2=spo2,
        bp_systolic=effective_sbp,
        bp_diastolic=effective_dbp,
        pulse=pulse,
        temperature=temperature,
        known_conditions=known_conditions,
        medications=medications
    )

    # 3. SAFETY ABSTENTION CHECK
    abstained, abstention_reason = evaluate_safety_abstention(
        complaint=complaint,
        spo2=spo2,
        bp_systolic=effective_sbp,
        pulse=pulse,
        temperature=temperature,
        model_conf=model_conf
    )

    if abstained:
        return {
            "encounter_id": patient_id,
            "tier": None,
            "tier_name": "Needs Nurse Review",
            "department": "Triage Review Desk",
            "abstained": True,
            "abstention_reason": f"SAFETY ABSTENTION: Cannot assess – send to nurse now. ({abstention_reason})",
            "reasons": [f"SAFETY ABSTENTION: Cannot assess – send to nurse now. ({abstention_reason})"],
            "warnings": validation_warnings,
            "red_flags": [],
            "is_escalated": False,
            "escalation_rule": None,
            "model_tier": model_tier,
            "model_confidence": model_conf,
            "cleaned_vitals": cleaned_vitals
        }

    # 4. HARD-RULE ESCALATION LAYER ON TOP (Can ONLY escalate, never downgrade!)
    forced_tier, escalation_rule, escalation_reasons, red_flags = evaluate_hard_escalations(
        spo2=spo2,
        bp_systolic=effective_sbp,
        bp_diastolic=effective_dbp,
        pulse=pulse,
        temperature=temperature,
        complaint=complaint
    )

    all_reasons = []

    # Case A: Hard-coded escalation triggered
    if forced_tier is not None:
        # Rule can only escalate: if model predicted higher urgency (e.g. L1 vs hard rule L2), keep L1
        if model_tier is not None and model_tier < forced_tier:
            final_tier = model_tier
            all_reasons.append(f"ML Model identified severe acuity: Level {model_tier} (Confidence: {model_conf*100:.1f}%)")
        else:
            final_tier = forced_tier

        all_reasons.extend(escalation_reasons)
        if model_tier is not None:
            all_reasons.append(f"Model proposal was Level {model_tier}; Hard-coded safety rule escalated to Level {final_tier}")

        rule_dept = recommend_department(complaint, age, gender, final_tier)
        if final_tier == 1:
            department = "Emergency/Casualty"
        elif rule_dept in ["Cardiology", "Emergency/Casualty"]:
            department = rule_dept
        elif model_dept:
            department = model_dept
        else:
            department = rule_dept

        return {
            "encounter_id": patient_id,
            "tier": final_tier,
            "tier_name": TRIAGE_LEVELS[final_tier]["name"],
            "department": department,
            "abstained": False,
            "abstention_reason": None,
            "reasons": all_reasons,
            "warnings": validation_warnings,
            "red_flags": red_flags,
            "is_escalated": True,
            "escalation_rule": escalation_rule,
            "model_tier": model_tier,
            "model_confidence": model_conf,
            "model_probs": [float(p) for p in model_probs] if model_probs is not None else None,
            "cleaned_vitals": cleaned_vitals
        }

    # Case B: No hard escalation triggered -> Use Model decision or Heuristic baseline
    if model_tier is not None:
        final_tier = model_tier
        if final_tier == 1:
            department = "Emergency/Casualty"
        elif model_dept:
            department = model_dept
        else:
            department = recommend_department(complaint, age, gender, final_tier)
        all_reasons.append(f"ML Triage Model classification: Level {final_tier} (Confidence: {model_conf*100:.1f}%)")
        all_reasons.append("Hard escalation guardrails checked: All vital signs and symptom keywords within non-critical parameters.")
    else:
        # Heuristic scoring fallback
        scored_tier, scoring_reasons = score_tier_without_hard_escalation(
            spo2=spo2,
            bp_systolic=effective_sbp,
            pulse=pulse,
            temperature=temperature,
            complaint=complaint
        )
        final_tier = scored_tier
        department = recommend_department(complaint, age, gender, final_tier)
        all_reasons.extend(scoring_reasons)

    return {
        "encounter_id": patient_id,
        "tier": final_tier,
        "tier_name": TRIAGE_LEVELS[final_tier]["name"],
        "department": department,
        "abstained": False,
        "abstention_reason": None,
        "reasons": all_reasons,
        "warnings": validation_warnings,
        "red_flags": red_flags,
        "is_escalated": False,
        "escalation_rule": None,
        "model_tier": model_tier,
        "model_confidence": model_conf,
        "model_probs": [float(p) for p in model_probs] if model_probs is not None else None,
        "cleaned_vitals": cleaned_vitals
    }


def validate_nurse_override(
    current_tier: Optional[int],
    new_tier: Optional[int],
    current_dept: str,
    new_dept: str,
    reason: str,
    is_escalated: bool,
    confirm_escalation_downgrade: bool = False
) -> Tuple[bool, Optional[str]]:
    """
    Validates nurse override request.
    Enforces mandatory clinical rationale and explicit confirmation when downgrading hard escalations.
    """
    clean_reason = (reason or "").strip()
    if len(clean_reason) < 5:
        return False, "Override rejected: A detailed clinical rationale (minimum 5 characters) is mandatory."

    if new_tier is not None and new_tier not in [1, 2, 3, 4, 5]:
        return False, "Override rejected: Invalid target tier specified."

    if new_dept and new_dept != "Triage Review Desk":
        canon_map = {d.lower(): d for d in DEPARTMENTS}
        canon_map.update({
            "pediatrics": "Paediatrics",
            "orthopedics": "Orthopaedics",
            "obstetrics & gynecology": "Obstetrics & Gynaecology",
            "emergency / resuscitation": "Emergency/Casualty",
            "respiratory / pulmonology": "Pulmonology/Chest & TB"
        })
        if new_dept.lower() not in canon_map:
            return False, f"Override rejected: Unknown department '{new_dept}'. Must be one of the supported hospital departments."

    if is_escalated and current_tier in [1, 2] and new_tier is not None and new_tier > current_tier:
        if not confirm_escalation_downgrade:
            return False, (
                "Safety Guardrail: This patient triggered a hard-coded clinical escalation. "
                "Downgrading requires an explicit secondary clinician confirmation."
            )

    return True, None
