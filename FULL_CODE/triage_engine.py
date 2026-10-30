"""CarePriority Gov - Clinical Triage & Guardrail Engine
======================================================
Pure functions, zero UI dependencies.
Architecture:
1. Data-entry validation & unit-error detection for vitals
   - BP of 0 = treated as missing, flagged for verification, stored as None.
   - Temperature values outside the primer's usable range are flagged for verification.
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
    temperature: Optional[float],
    temp_unit: str = "F"
) -> Dict[str, Optional[float]]:
    """
    Cleans vitals according to strict data entry rules:
    - BP of 0 = treated as missing, stored as None.
    - Missing values stay None, never coerced to 0.
    - Temperature is stored in Fahrenheit because existing encounter displays use °F.
    """
    sbp = None if (bp_systolic is None or bp_systolic == 0) else int(bp_systolic)
    dbp = None if (bp_diastolic is None or bp_diastolic == 0 or bp_systolic == 0) else int(bp_diastolic)
    temperature_f = temperature

    if temperature is not None and temp_unit and temp_unit.upper() == "C":
        if (
            VITALS_PLAUSIBILITY["temperature"]["min_celsius"]
            <= temperature
            <= VITALS_PLAUSIBILITY["temperature"]["max_celsius"]
        ):
            temperature_f = temperature * 9 / 5 + 32
        else:
            temperature_f = None

    return {
        "bp_systolic": sbp,
        "bp_diastolic": dbp,
        "spo2": spo2,
        "pulse": pulse,
        "temperature": temperature_f
    }


def validate_vitals(
    age: Optional[float],
    spo2: Optional[float],
    bp_systolic: Optional[int],
    bp_diastolic: Optional[int],
    pulse: Optional[int],
    temperature: Optional[float],
    temp_unit: str = "F",
    age_months: Optional[float] = None
) -> Tuple[bool, List[str], List[str]]:
    """
    Validates entered vitals for clinical plausibility and entry errors.
    Returns: (is_valid, error_messages, warning_messages)

    DATA-ENTRY RULES:
    - BP of 0 = treated as missing, flagged for verification, stored as None.
    - Temperature outside the primer's usable range requires verification.
    - Missing values stay None, never coerced to 0.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Age validation
    age_to_validate = age_months / 12 if age_months is not None else age
    if age_to_validate is not None:
        if (
            age_to_validate < VITALS_PLAUSIBILITY["age"]["min"]
            or age_to_validate > VITALS_PLAUSIBILITY["age"]["max"]
        ):
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
            errors.append(
                f"Systolic BP ({bp_systolic} mmHg) is outside the usable range (20–260 mmHg)."
            )

    if bp_diastolic is not None:
        if bp_diastolic == 0:
            warnings.append("Diastolic BP of 0 detected: Treated as missing/unobtainable and stored as None.")
        elif bp_diastolic < VITALS_PLAUSIBILITY["bp_diastolic"]["min"] or bp_diastolic > VITALS_PLAUSIBILITY["bp_diastolic"]["max"]:
            errors.append(
                f"Diastolic BP ({bp_diastolic} mmHg) is outside the usable range (20–260 mmHg)."
            )

    if bp_systolic is not None and bp_diastolic is not None and bp_systolic > 0 and bp_diastolic > 0:
        if bp_diastolic >= bp_systolic:
            errors.append(f"Diastolic BP ({bp_diastolic}) cannot be greater than or equal to Systolic BP ({bp_systolic}).")

    # 4. Pulse validation
    if pulse is not None:
        if pulse < VITALS_PLAUSIBILITY["pulse"]["min"] or pulse > VITALS_PLAUSIBILITY["pulse"]["max"]:
            errors.append(
                f"Pulse rate ({pulse} bpm) is outside the usable range (25–250 bpm)."
            )

    # 5. Temperature plausibility & unit error check
    if temperature is not None:
        unit = temp_unit.upper() if temp_unit else "F"
        min_c = VITALS_PLAUSIBILITY["temperature"]["min_celsius"]
        max_c = VITALS_PLAUSIBILITY["temperature"]["max_celsius"]

        # Explicit Celsius input check
        if unit == "C":
            if temperature < min_c or temperature > max_c:
                errors.append(
                    f"Temperature of {temperature}°C is outside the primer's usable range "
                    f"({min_c:.1f}°C–{max_c:.1f}°C) and requires verification."
                )
        else:
            # Fahrenheit input checks
            min_f = VITALS_PLAUSIBILITY["temperature"]["min_fahrenheit"]
            max_f = VITALS_PLAUSIBILITY["temperature"]["max_fahrenheit"]
            if min_c <= temperature <= max_c:
                errors.append(
                    f"Temperature of {temperature}° is in the Celsius range; verify the selected unit."
                )
            elif temperature < min_f or temperature > max_f:
                errors.append(
                    f"Temperature ({temperature}°F) is outside the primer's usable range "
                    f"({min_f:.1f}°F–{max_f:.1f}°F) and requires verification."
                )

    return (len(errors) == 0, errors, warnings)


def detect_symptom_matches(text: str, keyword_list: List[str]) -> List[str]:
    """
    Matches symptom keywords in free text while respecting clear local negation
    and the primer's chest-discomfort exclusions.
    """
    if not text:
        return []

    text_lower = text.lower()
    matches: List[str] = []
    chest_discomfort_terms = {
        "chest pain", "chest pressure", "chest heaviness", "chest tightness",
        "pressure in chest", "heaviness in chest", "tightness in chest",
        "छाती में दर्द", "सीने में दर्द", "छाती में जकड़न",
        "గుండె నొప్పి", "ఛాతీ నొప్పి", "ఛాతీలో మంట",
        "seene me dard", "seene mein dard", "chaati me dard",
        "chaati mein dard", "chati me dard", "chati mein dard",
        "chhati me dard", "dil me dard", "gunde noppi",
        "chhati noppi", "chaati noppi", "chathi noppi",
        "chhati lo manta",
    }

    def clause_start(position: int, include_comma: bool) -> int:
        separators = r"[.!?;\n,]" if include_comma else r"[.!?;\n]"
        separators += r"|\b(?:but|however|although|except|yet)\b"
        boundaries = list(re.finditer(separators, text_lower[:position]))
        return boundaries[-1].end() if boundaries else 0

    def is_negated(start: int, end: int, keyword: str) -> bool:
        # These phrases describe positive danger signs despite containing "not".
        if keyword.lower().startswith(("not ", "no ")):
            return False

        prefix = text_lower[clause_start(start, include_comma=True):start]
        prefix = prefix[-64:]
        negation = (
            r"(?:\bno\b|\bden(?:y|ies|ied)\b|\bwithout\b|\bnot\b|"
            r"\bnever\b|\bdoesn't\b|\bdoes\s+not\b|\bisn't\b|"
            r"\bis\s+not\b|\baren't\b|\bare\s+not\b|\bhas\s+no\b|"
            r"\bhave\s+no\b|\bfree\s+of\b|\bnegative\s+for\b|"
            r"\babsent\b)"
        )
        permitted_gap = (
            r"(?:(?:any|the|a|an|having|experiencing|reporting|"
            r"currently|present|of|longer|have|has)\s+){0,4}"
        )
        if re.search(negation + r"\s+" + permitted_gap + r"$", prefix):
            return True

        suffix_end_match = re.search(
            r"\b(?:but|however|although|except|yet)\b", text_lower[end:]
        )
        suffix_end = end + suffix_end_match.start() if suffix_end_match else len(text_lower)
        suffix = text_lower[end:suffix_end]
        return bool(re.match(r"^\s*(?:is\s+)?(?:absent|denied|not present)\b", suffix))

    def is_chest_discomfort_excluded(start: int, end: int) -> bool:
        # Commas are allowed within a clearly stated causal phrase.
        left = clause_start(start, include_comma=False)
        right_match = re.search(
            r"[.!?;\n]|\b(?:but|however|although|except|yet)\b",
            text_lower[end:],
        )
        right = end + right_match.start() if right_match else len(text_lower)
        context = text_lower[max(left, start - 40):min(right, end + 40)]
        injury_attribution = re.search(
            r"\b(?:pain|pressure|heaviness|tightness)\b"
            r".{0,32}\b(?:caused by|due to|because of|from|related to|attributed to)\b"
            r".{0,20}\b(?:injury|trauma|fall|blow|impact)\b"
            r"|\b(?:injury|trauma|fall|blow|impact)\b"
            r".{0,24}\b(?:causes?|caused|triggered|triggers|resulted in)\b"
            r".{0,24}\b(?:chest\s+)?(?:pain|pressure|heaviness|tightness)\b",
            context,
        )
        cough_attribution = re.search(
            r"\b(?:pain|pressure|heaviness|tightness)\b"
            r".{0,32}\b(?:only\s+)?(?:when|while|on)\s+cough(?:ing)?\b"
            r"|\b(?:pain|pressure|heaviness|tightness)\b"
            r".{0,24}\b(?:caused by|due to|from|because of)\b"
            r".{0,16}\bcough(?:ing)?\b"
            r"|\bcough(?:ing)?\b.{0,24}\b(?:causes?|caused|triggers?|triggered)\b"
            r".{0,24}\b(?:chest\s+)?(?:pain|pressure|heaviness|tightness)\b",
            context,
        )
        meal_attribution = re.search(
            r"\b(?:pain|pressure|heaviness|tightness)\b"
            r".{0,32}\b(?:after|with|during)\s+(?:a\s+)?"
            r"(?:meals?|eating|food)\b"
            r"|\b(?:pain|pressure|heaviness|tightness)\b"
            r".{0,24}\b(?:caused by|due to|from|because of)\b"
            r".{0,16}\b(?:meals?|eating|food)\b"
            r"|\b(?:meals?|eating|food)\b.{0,24}\b"
            r"(?:causes?|caused|triggers?|triggered)\b"
            r".{0,24}\b(?:chest\s+)?(?:pain|pressure|heaviness|tightness)\b",
            context,
        )
        return bool(injury_attribution or cough_attribution or meal_attribution)

    for kw in keyword_list:
        kw_clean = kw.strip()
        if not kw_clean:
            continue
        kw_lower = kw_clean.lower()
        chest_discomfort = kw_lower in chest_discomfort_terms
        start = text_lower.find(kw_lower)
        while start >= 0:
            end = start + len(kw_lower)
            if not is_negated(start, end, kw_clean) and not (
                chest_discomfort and is_chest_discomfort_excluded(start, end)
            ):
                matches.append(kw_clean)
                break
            start = text_lower.find(kw_lower, start + 1)

    return list(dict.fromkeys(matches))


_STAGE1_CANONICAL_KEYWORD_EXCLUSIONS = {
    "L1": {
        "active seizure",
        "convulsing now",
        "status epilepticus",
        "bleeding that will not stop",
    },
    "L2": {
        "stroke",
        "paralysis",
        "facial droop",
        "slurred speech",
        "weakness in arm",
        "shortness of breath",
        "dyspnea",
        "cannot speak full sentences",
        "struggling to breathe",
        "chest indrawing",
        "choking on a foreign body",
        "seizure",
        "fit",
        "fits",
        "convulsion",
        "severe bleeding",
        "profuse bleeding",
        "coughing blood",
        "hemoptysis",
        "vomiting blood",
        "hematemesis",
        "altered mental state",
    },
}


def detect_stage1_red_flag_keyword_matches(
    text: Optional[str], level: str
) -> List[str]:
    """Return legacy matches not replaced by Stage 1 contextual code rules."""
    category = "L1_SYMPTOMS" if level == "L1" else "L2_SYMPTOMS"
    excluded = _STAGE1_CANONICAL_KEYWORD_EXCLUSIONS.get(level, set())
    return [
        match
        for match in detect_symptom_matches(text or "", RED_FLAG_KEYWORDS[category]["keywords"])
        if match.lower() not in excluded
    ]


def detect_stage1_canonical_red_flags(
    text: Optional[str],
) -> Tuple[List[str], List[str], bool]:
    """Detect the five Stage 1 primer red flags with context and local negation.

    Returns canonical codes, the codes that independently meet Level 1 criteria,
    and whether severe respiratory distress is present for the existing
    SpO2-plus-distress escalation.
    """
    complaint = text or ""
    flags: List[str] = []

    def has_any(phrases: List[str]) -> bool:
        return bool(detect_symptom_matches(complaint, phrases))

    # The primer's FAST features must have sudden onset.
    stroke_phrases = [
        "sudden face droop", "sudden facial droop", "sudden drooping of face",
        "face suddenly drooping", "face drooped suddenly",
        "sudden one-sided arm weakness", "sudden one sided arm weakness",
        "sudden weakness in one arm", "sudden weakness of one arm",
        "sudden weakness on one side of the arm",
        "sudden weakness in left arm", "sudden weakness in right arm",
        "sudden weakness of left arm", "sudden weakness of right arm",
        "sudden one-sided leg weakness", "sudden one sided leg weakness",
        "sudden weakness in one leg", "sudden weakness of one leg",
        "sudden weakness on one side of the leg",
        "sudden weakness in left leg", "sudden weakness in right leg",
        "sudden weakness of left leg", "sudden weakness of right leg",
        "sudden one-sided arm or leg weakness",
        "sudden slurred speech", "speech suddenly slurred",
        "suddenly slurred speech", "sudden absent speech",
        "suddenly unable to speak", "sudden inability to speak",
    ]
    if has_any(stroke_phrases):
        flags.append("RF_STROKE_SIGNS")

    respiratory_phrases = [
        "cannot speak full sentences", "can't speak full sentences",
        "struggling to breathe", "choking on a foreign body",
        "severe difficulty breathing", "severe difficulty in breathing",
        "severe breathing difficulty",
    ]
    child_indrawing = has_any([
        "sick child with chest indrawing",
        "sick child has chest indrawing",
        "sick child having chest indrawing",
        "chest indrawing in a sick child",
    ])
    respiratory_distress = has_any(respiratory_phrases) or child_indrawing
    if respiratory_distress:
        flags.append("RF_RESP_DISTRESS")

    altered_sensorium = has_any([
        "new confusion", "newly confused", "new onset confusion",
        "new drowsiness", "newly drowsy", "new onset drowsiness",
        "sleepiness after a fit", "sleepy after a fit",
        "sleepiness after a seizure", "sleepy after a seizure",
        "not recognising family", "not recognizing family",
        "does not recognise family", "does not recognize family",
        "doesn't recognise family", "doesn't recognize family",
        "unresponsive", "unconscious",
    ])
    if altered_sensorium:
        flags.append("RF_ALTERED_SENSORIUM")
    level1_flags = (
        ["RF_ALTERED_SENSORIUM"]
        if has_any(["unresponsive", "unconscious"])
        else []
    )

    active_seizure = has_any([
        "convulsing now", "currently convulsing", "actively convulsing",
        "active seizure", "status epilepticus", "currently having a seizure",
        "having a seizure now", "seizure happening now",
        "seizures happening now", "seizure now", "seizures now",
        "currently having a fit", "having a fit now", "fit happening now",
        "fits happening now", "fits now", "convulsion happening now",
        "convulsions happening now", "convulsions now",
    ])
    recent_seizure = has_any([
        "seizure within the last few hours", "seizure in the last few hours",
        "seizures within the last few hours", "seizures in the last few hours",
        "seizure within last few hours", "seizure in last few hours",
        "fit within the last few hours", "fit in the last few hours",
        "fits within the last few hours", "fits in the last few hours",
        "fit within last few hours", "fit in last few hours",
        "convulsion within the last few hours",
        "convulsion in the last few hours",
        "convulsions within the last few hours",
        "convulsions in the last few hours",
        "had a seizure a few hours ago", "had a fit a few hours ago",
        "had convulsions a few hours ago",
        "seizure a few hours ago", "fit a few hours ago",
        "seizure 2 hours ago", "had a seizure 2 hours ago",
        "fit 2 hours ago", "had a fit 2 hours ago",
        "convulsion 2 hours ago", "had a convulsion 2 hours ago",
        "seizure two hours ago", "had a seizure two hours ago",
        "fit two hours ago", "had a fit two hours ago",
        "convulsion two hours ago", "had a convulsion two hours ago",
    ])
    if active_seizure or recent_seizure:
        flags.append("RF_SEIZURE")
    if active_seizure:
        level1_flags.append("RF_SEIZURE")

    bleeding_l2 = has_any([
        "vomiting blood", "vomited blood", "blood in vomit", "hematemesis",
        "black tarry stool", "black tarry stools", "tarry black stool",
        "tarry black stools", "coughing up large amounts of blood",
        "coughing up a large amount of blood",
        "coughing up large amount of blood",
        "coughing up a lot of blood",
        "coughing up lots of blood",
        "large amounts of blood when coughing",
    ])
    wound_bleeding_l1 = has_any([
        "bleeding that will not stop",
        "wound bleeding will not stop",
        "wound bleeding that will not stop",
        "bleeding from wound will not stop",
        "bleeding from wound that will not stop",
        "bleeding from the wound will not stop",
        "bleeding from the wound that will not stop",
        "wound won't stop bleeding",
        "wound will not stop bleeding",
        "wound bleeding won't stop",
        "wound bleeding is not stopping",
        "bleeding from wound is not stopping",
        "bleeding from the wound is not stopping",
    ])
    if bleeding_l2 or wound_bleeding_l1:
        flags.append("RF_SEVERE_BLEEDING")
    if wound_bleeding_l1:
        level1_flags.append("RF_SEVERE_BLEEDING")

    return (
        list(dict.fromkeys(flags)),
        list(dict.fromkeys(level1_flags)),
        respiratory_distress,
    )


def evaluate_safety_abstention(
    complaint: Optional[str],
    spo2: Optional[float],
    bp_systolic: Optional[int],
    pulse: Optional[int],
    temperature: Optional[float],
    model_conf: Optional[float] = None,
    bp_diastolic: Optional[int] = None
) -> Tuple[bool, Optional[str]]:
    """
    Safety Abstention Guardrail:
    If the complaint or all primary vital signs are missing, the complaint is
    uninformative, the record is contradictory, or model confidence is low, the
    system MUST abstain and flag for immediate nurse review.
    Abstention must NEVER default to Level 5.
    """
    complaint_clean = (complaint or "").strip()
    has_key_vitals = (spo2 is not None) or (pulse is not None) or (bp_systolic is not None)

    if not has_key_vitals:
        return True, "No primary respiratory or cardiovascular vital signs recorded. Nurse assessment is required."

    normalized_complaint = complaint_clean.lower()
    uninformative_complaints = {
        "", "-", "--", "?", "??", "doctor ko dikhana hai", "language not understood"
    }
    complaint_unusable = (
        normalized_complaint in uninformative_complaints
        or len(complaint_clean) < 4
        or not any(character.isalpha() for character in complaint_clean)
    )
    plausible = VITALS_PLAUSIBILITY
    usable_vital_count = sum([
        spo2 is not None and plausible["spo2"]["min"] <= spo2 <= plausible["spo2"]["max"],
        pulse is not None and plausible["pulse"]["min"] <= pulse <= plausible["pulse"]["max"],
        bp_systolic is not None and bp_diastolic is not None
        and bp_systolic > 0 and bp_diastolic > 0
        and plausible["bp_systolic"]["min"] <= bp_systolic <= plausible["bp_systolic"]["max"]
        and plausible["bp_diastolic"]["min"] <= bp_diastolic <= plausible["bp_diastolic"]["max"],
        temperature is not None
        and plausible["temperature"]["min_fahrenheit"] <= temperature
        <= plausible["temperature"]["max_fahrenheit"],
    ])
    if complaint_unusable and usable_vital_count <= 1:
        return True, "Unusable complaint with at most one usable vital sign. Nurse assessment is required."

    # Condition 4: Contradiction: BP 0 but patient has pulse > 50 without emergency tag
    if bp_systolic == 0 and pulse is not None and pulse > 50:
        return True, "Contradictory hemodynamic record: Systolic BP recorded as 0 mmHg with active pulse. Requires immediate manual verification."

    # Low confidence is a review trigger even when some vitals are available.
    if model_conf is not None and model_conf < ABSTENTION_CONDITIONS["low_model_confidence"]["threshold"]:
        return True, "Low triage model confidence. Send to triage nurse for manual assessment."

    return False, None


def evaluate_hard_escalations(
    spo2: Optional[float],
    bp_systolic: Optional[int],
    bp_diastolic: Optional[int],
    pulse: Optional[int],
    temperature: Optional[float],
    complaint: Optional[str],
    age: Optional[float] = None,
    age_months: Optional[float] = None,
    pregnant: Optional[bool] = None,
    pregnancy_stage: Optional[str] = None,
    recently_delivered: bool = False,
    known_conditions: Optional[str] = None,
    medications: Optional[str] = None
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
    complaint_lower = complaint_str.lower()
    years = age_months / 12 if age_months is not None else age
    months = age_months if age_months is not None else (age * 12 if age is not None else None)
    if years is not None and not (
        VITALS_PLAUSIBILITY["age"]["min"] <= years <= VITALS_PLAUSIBILITY["age"]["max"]
    ):
        years = None
        months = None
    if months is not None and not (0 <= months <= VITALS_PLAUSIBILITY["age"]["max"] * 12):
        years = None
        months = None
    config = VITALS_PLAUSIBILITY
    spo2 = (
        spo2 if spo2 is not None
        and config["spo2"]["min"] <= spo2 <= config["spo2"]["max"]
        else None
    )
    pulse = (
        pulse if pulse is not None
        and config["pulse"]["min"] <= pulse <= config["pulse"]["max"]
        else None
    )
    bp_systolic = (
        bp_systolic if bp_systolic is not None and bp_systolic > 0
        and config["bp_systolic"]["min"] <= bp_systolic <= config["bp_systolic"]["max"]
        else None
    )
    bp_diastolic = (
        bp_diastolic if bp_diastolic is not None and bp_diastolic > 0
        and config["bp_diastolic"]["min"] <= bp_diastolic <= config["bp_diastolic"]["max"]
        else None
    )
    temp_min_f = config["temperature"]["min_fahrenheit"]
    temp_max_f = config["temperature"]["max_fahrenheit"]
    temperature = (
        temperature if temperature is not None and temp_min_f <= temperature <= temp_max_f
        else None
    )
    l1_vitals = HARD_ESCALATION_VITALS["L1"]
    l2_vitals = HARD_ESCALATION_VITALS["L2"]

    canonical_flags, canonical_level1_flags, respiratory_distress = (
        detect_stage1_canonical_red_flags(complaint_str)
    )
    if canonical_flags:
        forced_tier = 2
        rule_name = "Primer canonical red-flag escalation"
        red_flags.extend(canonical_flags)
        reasons.append(
            "ESCALATION RULE TRIGGERED: Canonical primer red flag detected "
            f"({', '.join(canonical_flags)})"
        )

    # Apply the context-dependent pregnancy red flags before vital checks.
    pregnancy_context = pregnant is True or recently_delivered
    stage = (pregnancy_stage or "").strip().lower()
    over_20_weeks = stage in {"more than 20 weeks", "third trimester"}
    in_third_trimester = stage == "third trimester"
    pregnancy_flags: List[str] = []
    if pregnancy_context and (
        "vaginal bleeding" in complaint_lower
        or "bleeding in pregnancy" in complaint_lower
        or (recently_delivered and any(term in complaint_lower for term in (
            "heavy bleeding after delivery", "heavy bleeding after birth", "postpartum bleeding"
        )))
    ):
        pregnancy_flags.append("RF_PREGNANCY_BLEEDING")
    eclampsia_symptoms = (
        "severe headache", "blurred vision", "upper-abdominal pain",
        "upper abdominal pain", "face swelling", "hand swelling", "swelling of face",
        "swelling of hands", "face/hand swelling", "fits", "convulsion", "seizure"
    )
    if pregnancy_context and (recently_delivered or (pregnant is True and over_20_weeks)) \
            and any(term in complaint_lower for term in eclampsia_symptoms):
        pregnancy_flags.append("RF_ECLAMPSIA_SIGNS")
    fetal_movement_terms = (
        "reduced fetal movement", "absent fetal movement", "baby not moving",
        "no baby movement", "reduced baby movements", "absent baby movements"
    )
    if pregnant is True and in_third_trimester and any(
        term in complaint_lower for term in fetal_movement_terms
    ):
        pregnancy_flags.append("RF_REDUCED_FETAL_MOVEMENT")
    if pregnancy_flags:
        forced_tier = 2
        rule_name = "Pregnancy-related red-flag escalation"
        red_flags.extend(pregnancy_flags)
        reasons.append(
            "ESCALATION RULE TRIGGERED: Primer pregnancy red flag detected "
            f"({', '.join(pregnancy_flags)})"
        )

    # Level 1 symptom keywords. Contextual criteria are evaluated separately below.
    l1_matches = detect_stage1_red_flag_keyword_matches(complaint_str, "L1")
    if l1_matches or canonical_level1_flags:
        forced_tier = 1
        rule_name = RED_FLAG_KEYWORDS["L1_SYMPTOMS"]["rule_name"]
        red_flags.extend(l1_matches)
        level1_evidence = l1_matches + canonical_level1_flags
        reasons.append(
            "ESCALATION RULE TRIGGERED: Critical resuscitation evidence detected "
            f"({', '.join(level1_evidence)})"
        )

    if spo2 is not None and spo2 < l1_vitals["spo2_critical"]:
        forced_tier = 1
        rule_name = "Critical hypoxia (SpO2 < 85%)"
        reasons.append(
            f"ESCALATION RULE TRIGGERED: Profound hypoxemia (SpO2 {spo2}% < 85%)"
        )

    if spo2 is not None and spo2 < l1_vitals["spo2_with_distress"] and respiratory_distress:
        forced_tier = 1
        rule_name = "Very low oxygen with severe respiratory distress"
        reasons.append(
            f"ESCALATION RULE TRIGGERED: SpO2 {spo2}% < 90% with severe respiratory distress"
        )

    if months is not None and years is not None and bp_systolic is not None:
        if years >= 10:
            hypotension_limit = l1_vitals["adult_sbp_shock"]
        elif months < 1:
            hypotension_limit = l1_vitals["infant_under_1_month_sbp_shock"]
        elif months < 12:
            hypotension_limit = l1_vitals["infant_under_12_months_sbp_shock"]
        else:
            hypotension_limit = (
                l1_vitals["child_sbp_shock_base"]
                + l1_vitals["child_sbp_shock_per_year"] * years
            )
        if bp_systolic < hypotension_limit:
            forced_tier = 1
            rule_name = f"Age-specific severe hypotension (SBP {bp_systolic} mmHg)"
            reasons.append(
                f"ESCALATION RULE TRIGGERED: SBP {bp_systolic} mmHg below "
                f"age-specific Level 1 limit {hypotension_limit:.1f} mmHg"
            )

    if years is not None and years < 8 and pulse is not None \
            and pulse < l1_vitals["child_pulse_bradycardia"]:
        forced_tier = 1
        rule_name = f"Child bradycardia (Pulse {pulse} bpm)"
        reasons.append(
            f"ESCALATION RULE TRIGGERED: Pulse {pulse} bpm < 60 bpm in a child under 8 years"
        )

    known_text = (known_conditions or "").lower()
    medication_text = (medications or "").lower()
    poisoning = any(term in complaint_lower for term in ("pesticide", "insecticide", "poison"))
    if poisoning and any(term in complaint_lower for term in ("frothing", "heavy secretions")):
        forced_tier = 1
        rule_name = "Poisoning requiring immediate life-saving drug"
        reasons.append(
            "ESCALATION RULE TRIGGERED: Pesticide/poison exposure with frothing or heavy secretions"
        )
    diabetic_context = any(
        term in known_text or term in medication_text
        for term in ("diabet", "insulin", "glimepiride", "sugar tablet")
    )
    if diabetic_context and any(term in complaint_lower for term in ("low sugar", "hypogly")) \
            and any(term in complaint_lower for term in ("confused", "drowsy")):
        forced_tier = 1
        rule_name = "Hypoglycaemia requiring immediate life-saving treatment"
        reasons.append(
            "ESCALATION RULE TRIGGERED: Diabetic with low sugar and confusion/drowsiness"
        )
    shock_description = (
        any(term in complaint_lower for term in ("collapse", "collapsed"))
        and "cold clammy" in complaint_lower
        and any(
            term in complaint_lower
            for term in ("weak pulse", "weak and fast pulse", "weak fast pulse")
        )
    )
    severe_allergy = (
        "throat closing" in complaint_lower
        or (
            any(term in complaint_lower for term in ("allergic", "allergy"))
            and any(term in complaint_lower for term in ("collapse", "collapsed"))
        )
    )
    if shock_description or severe_allergy:
        forced_tier = 1
        rule_name = "Shock or severe allergic reaction"
        reasons.append(
            "ESCALATION RULE TRIGGERED: Primer Level 1 shock/allergy symptom combination"
        )

    # If already forced to Level 1, return immediately (cannot escalate higher)
    if forced_tier == 1:
        return 1, rule_name or "Level 1 Resuscitation", reasons, red_flags

    # 3. Level 2 Hard Symptom Keywords (Chest pain, stroke, poisoning, snake bite, etc.)
    l2_matches = detect_stage1_red_flag_keyword_matches(complaint_str, "L2")
    if l2_matches:
        forced_tier = 2
        rule_name = RED_FLAG_KEYWORDS["L2_SYMPTOMS"]["rule_name"]
        red_flags.extend(l2_matches)
        reasons.append(f"ESCALATION RULE TRIGGERED: Emergent red-flag keyword ({', '.join(l2_matches)})")

    if spo2 is not None and spo2 < l2_vitals["spo2_hypoxia"]:
        forced_tier = 2
        rule_name = f"Mandatory Hypoxia Escalation (SpO2 {spo2}% < 92%)"
        reasons.append(f"ESCALATION RULE TRIGGERED: Hypoxemia (SpO2 {spo2}% < 92%)")

    if years is not None and years >= 10 and bp_systolic is not None \
            and bp_systolic <= l2_vitals["adult_sbp_hypotension"]:
        forced_tier = 2
        rule_name = f"Age-specific hypotension (SBP {bp_systolic} mmHg)"
        reasons.append(
            f"ESCALATION RULE TRIGGERED: SBP {bp_systolic} mmHg <= 90 mmHg at age 10 years or older"
        )

    if years is not None and years >= 16 and bp_systolic is not None \
            and bp_systolic >= l2_vitals["adult_sbp_crisis"]:
        forced_tier = 2
        rule_name = f"Adult systolic hypertension danger zone (SBP {bp_systolic} mmHg)"
        reasons.append(
            f"ESCALATION RULE TRIGGERED: SBP {bp_systolic} mmHg >= 220 mmHg at age 16 years or older"
        )

    if pregnant is True and bp_systolic is not None \
            and bp_systolic >= l2_vitals["pregnant_sbp_crisis"]:
        forced_tier = 2
        rule_name = f"Pregnancy blood-pressure danger zone (SBP {bp_systolic} mmHg)"
        reasons.append(
            f"ESCALATION RULE TRIGGERED: Pregnant patient with SBP >= 160 mmHg"
        )

    if pregnant is True and bp_diastolic is not None \
            and bp_diastolic >= l2_vitals["pregnant_dbp_crisis"]:
        forced_tier = 2
        rule_name = f"Pregnancy blood-pressure danger zone (DBP {bp_diastolic} mmHg)"
        reasons.append(
            "ESCALATION RULE TRIGGERED: Pregnant patient with DBP >= 110 mmHg"
        )

    if pulse is not None and months is not None and years is not None:
        pulse_limit: Optional[float] = None
        is_pulse_danger = False
        if months < 3:
            pulse_limit = l2_vitals["pulse_under_3_months_high"]
            is_pulse_danger = pulse > pulse_limit
        elif months < 36:
            pulse_limit = l2_vitals["pulse_3_to_under_36_months_high"]
            is_pulse_danger = pulse > pulse_limit
        elif years < 8:
            pulse_limit = l2_vitals["pulse_3_to_under_8_years_high"]
            is_pulse_danger = pulse > pulse_limit
        elif years < 16:
            high_limit = l2_vitals["pulse_8_to_under_16_years_high"]
            low_limit = l2_vitals["pulse_8_to_under_16_years_low"]
            is_pulse_danger = pulse > high_limit or pulse <= low_limit
            pulse_limit = high_limit
        else:
            high_limit = l2_vitals["pulse_16_years_and_over_high"]
            low_limit = l2_vitals["pulse_16_years_and_over_low"]
            is_pulse_danger = pulse >= high_limit or pulse <= low_limit
            pulse_limit = high_limit
        if is_pulse_danger:
            forced_tier = 2
            rule_name = f"Age-specific heart-rate danger zone (Pulse {pulse} bpm)"
            reasons.append(
                f"ESCALATION RULE TRIGGERED: Pulse {pulse} bpm is outside the "
                f"primer's age-specific danger-zone threshold ({pulse_limit} bpm)"
            )

    temperature_c = (temperature - 32) * 5 / 9 if temperature is not None else None
    if temperature_c is not None and temperature_c <= l2_vitals["temperature_danger_celsius"]:
        forced_tier = 2
        rule_name = f"Hypothermia (Temp {temperature_c:.1f}°C)"
        reasons.append(
            f"ESCALATION RULE TRIGGERED: Temperature {temperature_c:.1f}°C <= 35.0°C"
        )
    if months is not None and months < 3 and temperature_c is not None:
        infant_temp_danger = (
            temperature_c >= l2_vitals["infant_temperature_fever_celsius"]
            or temperature_c < l2_vitals["infant_temperature_hypothermia_celsius"]
        )
        if infant_temp_danger:
            forced_tier = 2
            rule_name = "Infant under 3 months temperature danger zone"
            if temperature_c >= l2_vitals["infant_temperature_fever_celsius"]:
                red_flags.append("RF_HIGH_FEVER_INFANT")
                reasons.append(
                    f"ESCALATION RULE TRIGGERED: Infant under 3 months with temperature "
                    f"{temperature_c:.1f}°C >= 38.0°C"
                )
            else:
                reasons.append(
                    f"ESCALATION RULE TRIGGERED: Infant under 3 months with temperature "
                    f"{temperature_c:.1f}°C < 36.0°C"
                )

    if forced_tier == 2:
        return 2, rule_name or "Level 2 Emergent Escalation", reasons, red_flags

    return None, None, reasons, red_flags


def score_tier_without_hard_escalation(
    spo2: Optional[float],
    bp_systolic: Optional[int],
    pulse: Optional[int],
    temperature: Optional[float],
    complaint: Optional[str],
    age: Optional[float] = None,
    age_months: Optional[float] = None
) -> Tuple[int, List[str]]:
    """
    Transparent clinical scoring heuristics for Levels 3, 4, 5.
    Used as baseline fallback when ML model is not available or for comparative evaluation.
    """
    reasons: List[str] = []
    complaint_str = (complaint or "").lower()

    resolved_age_months = age_months if age_months is not None else (age * 12 if age is not None else None)
    temperature_c = (temperature - 32) * 5 / 9 if temperature is not None else None
    has_fever = (
        temperature_c is not None and temperature_c >= 39.1
        and resolved_age_months is not None and resolved_age_months >= 3
    )
    has_mild_tachy = (
        age is not None and age >= 16 and pulse is not None and 111 <= pulse <= 130
    )
    has_mild_spo2_dip = (spo2 is not None and 92 <= spo2 <= 93)

    l3_matches = detect_symptom_matches(complaint_str, SECONDARY_KEYWORDS["L3_URGENT"])
    l4_matches = detect_symptom_matches(complaint_str, SECONDARY_KEYWORDS["L4_LESS_URGENT"])
    l5_matches = detect_symptom_matches(complaint_str, SECONDARY_KEYWORDS["L5_NON_URGENT"])

    if l3_matches:
        reasons.append(f"Urgent symptom keyword matched: {', '.join(l3_matches)}")
        return 3, reasons

    if sum([bool(has_fever), bool(has_mild_tachy), bool(has_mild_spo2_dip)]) >= 2:
        vital_details = []
        if has_fever: vital_details.append(f"Fever {temperature_c:.1f}°C")
        if has_mild_tachy: vital_details.append(f"Pulse {pulse} bpm")
        if has_mild_spo2_dip: vital_details.append(f"SpO2 {spo2}%")
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
    age: Optional[float],
    gender: Optional[str],
    tier: Optional[int],
    age_months: Optional[float] = None,
    pregnant: Optional[bool] = None,
    recently_delivered: bool = False
) -> str:
    """Recommends clinical department based on transparent rule priority across 13 canonical departments."""
    if tier in [1, 2]:
        return "Emergency/Casualty"

    complaint_str = (complaint or "").lower()

    # 1. Specialized physical/organ injuries checked first (Dental, Ophthalmology, ENT, Orthopaedics)
    organ_specialties = [
        "Dental", "Ophthalmology", "ENT", "Orthopaedics", "Dermatology", "General Surgery"
    ]
    for spec in organ_specialties:
        rule = DEPARTMENT_RULES.get(spec)
        if rule and "keywords" in rule:
            matches = detect_symptom_matches(complaint_str, rule["keywords"])
            if matches:
                return spec

    # Pregnancy-related problems route to obstetrics before the pediatric medical rule.
    obgyn_rule = DEPARTMENT_RULES.get("Obstetrics & Gynaecology") or DEPARTMENT_RULES.get(
        "Obstetrics & Gynecology", {}
    )
    if pregnant is True or recently_delivered:
        pregnancy_problem_matches = detect_symptom_matches(
            complaint_str,
            obgyn_rule.get("keywords", []) + [
                "vomiting", "urine burning", "burning urine", "dysuria",
                "mildly raised bp", "high bp", "blood pressure"
            ]
        )
        if pregnancy_problem_matches:
            return "Obstetrics & Gynaecology"

    # Children under 12 with medical complaints route to Paediatrics.
    paed_rule = DEPARTMENT_RULES.get("Paediatrics") or DEPARTMENT_RULES.get("Pediatrics", {})
    age_in_years = age_months / 12 if age_months is not None else age
    if age_in_years is not None and age_in_years < paed_rule.get("max_age", 12):
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

    # 4. Obstetrics & Gynaecology (legacy text/gender routing)
    if gender and gender.lower() in ["female", "f"]:
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
    age: Optional[float],
    gender: Optional[str],
    spo2: Optional[float],
    bp_systolic: Optional[int],
    bp_diastolic: Optional[int],
    pulse: Optional[int],
    temperature: Optional[float],
    complaint: Optional[str],
    known_conditions: Optional[str] = None,
    medications: Optional[str] = None,
    temp_unit: str = "F",
    age_months: Optional[float] = None,
    pregnant: Optional[bool] = None,
    pregnancy_stage: Optional[str] = None,
    recently_delivered: bool = False
) -> Dict[str, Any]:
    """
    Complete hybrid clinical triage pipeline:
    1. Hard emergency rules (always precede validation, model inference, and abstention)
    2. Validation & data-entry checks (BP 0, Celsius errors, plausible ranges)
    3. Model inference and safety abstention for cases without an emergency trigger
    5. Explainability tree & red-flag chips.
    """
    temperature_is_valid, _, _ = validate_vitals(
        age=None,
        spo2=None,
        bp_systolic=None,
        bp_diastolic=None,
        pulse=None,
        temperature=temperature,
        temp_unit=temp_unit
    )
    temperature_f = None
    temperature_c = None
    if temperature is not None and temperature_is_valid:
        if temp_unit and temp_unit.upper() == "C":
            temperature_c = temperature
            temperature_f = temperature * 9 / 5 + 32
        else:
            temperature_f = temperature
            temperature_c = (temperature - 32) * 5 / 9

    cleaned_vitals = clean_vitals_for_storage(
        bp_systolic, bp_diastolic, spo2, pulse, temperature, temp_unit
    )
    effective_sbp = cleaned_vitals["bp_systolic"]
    effective_dbp = cleaned_vitals["bp_diastolic"]

    # Emergency rules are checked before validation or missing-data abstention so
    # incomplete or malformed ancillary fields cannot suppress a red-flag trigger.
    forced_tier, escalation_rule, escalation_reasons, red_flags = evaluate_hard_escalations(
        spo2=spo2,
        bp_systolic=effective_sbp,
        bp_diastolic=effective_dbp,
        pulse=pulse,
        temperature=temperature_f,
        complaint=complaint,
        age=age,
        age_months=age_months,
        pregnant=pregnant,
        pregnancy_stage=pregnancy_stage,
        recently_delivered=recently_delivered,
        known_conditions=known_conditions,
        medications=medications
    )

    # 1. Validation
    is_valid, validation_errors, validation_warnings = validate_vitals(
        age=age,
        spo2=spo2,
        bp_systolic=bp_systolic,
        bp_diastolic=bp_diastolic,
        pulse=pulse,
        temperature=temperature,
        temp_unit=temp_unit,
        age_months=age_months
    )

    if forced_tier is not None:
        model_tier = None
        model_conf = None
        model_probs = None
        model_dept = None
        if is_valid:
            model_tier, model_conf, model_probs, model_dept = predict_ml_triage(
                complaint=complaint or "",
                age=age,
                gender=gender,
                spo2=spo2,
                bp_systolic=effective_sbp,
                bp_diastolic=effective_dbp,
                pulse=pulse,
                temperature=temperature_c,
                known_conditions=known_conditions,
                medications=medications
            )
        final_tier = min(forced_tier, model_tier) if model_tier is not None else forced_tier
        reasons = list(escalation_reasons)
        if model_tier is not None and model_tier < forced_tier:
            reasons.insert(
                0,
                f"ML Model identified more severe acuity: Level {model_tier}; emergency rule remains active."
            )

        department = "Emergency/Casualty"
        return {
            "encounter_id": patient_id,
            "tier": final_tier,
            "tier_name": TRIAGE_LEVELS[final_tier]["name"],
            "department": department,
            "abstained": False,
            "abstention_reason": None,
            "reasons": reasons,
            "warnings": validation_warnings + [
                f"Input verification required: {error}" for error in validation_errors
            ],
            "red_flags": red_flags,
            "is_escalated": True,
            "escalation_rule": escalation_rule,
            "model_tier": model_tier,
            "model_confidence": model_conf,
            "model_probs": [float(p) for p in model_probs] if model_probs is not None else None,
            "cleaned_vitals": cleaned_vitals
        }

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
            "model_probs": None,
            "cleaned_vitals": cleaned_vitals
        }

    # 2. Predict non-emergency cases using the model's Celsius temperature feature.
    model_tier, model_conf, model_probs, model_dept = predict_ml_triage(
        complaint=complaint or "",
        age=age,
        gender=gender,
        spo2=spo2,
        bp_systolic=effective_sbp,
        bp_diastolic=effective_dbp,
        pulse=pulse,
        temperature=temperature_c,
        known_conditions=known_conditions,
        medications=medications
    )

    # 3. SAFETY ABSTENTION CHECK
    abstained, abstention_reason = evaluate_safety_abstention(
        complaint=complaint,
        spo2=spo2 if spo2 is not None and 50 <= spo2 <= 100 else None,
        bp_systolic=bp_systolic,
        pulse=pulse,
        temperature=temperature_f,
        model_conf=model_conf,
        bp_diastolic=bp_diastolic
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

    all_reasons = []

    # Case B: No hard escalation triggered -> Use Model decision or Heuristic baseline
    if model_tier is not None:
        final_tier = model_tier
        if final_tier in [1, 2]:
            department = "Emergency/Casualty"
        elif model_dept:
            department = model_dept
        else:
            department = recommend_department(
                complaint, age, gender, final_tier, age_months, pregnant, recently_delivered
            )
        all_reasons.append(f"ML Triage Model classification: Level {final_tier} (Confidence: {model_conf*100:.1f}%)")
        all_reasons.append("Hard escalation guardrails checked: All vital signs and symptom keywords within non-critical parameters.")
    else:
        # Heuristic scoring fallback
        scored_tier, scoring_reasons = score_tier_without_hard_escalation(
            spo2=spo2,
            bp_systolic=effective_sbp,
            pulse=pulse,
            temperature=temperature_f,
            complaint=complaint,
            age=age,
            age_months=age_months
        )
        final_tier = scored_tier
        department = recommend_department(
            complaint, age, gender, final_tier, age_months, pregnant, recently_delivered
        )
        all_reasons.extend(scoring_reasons)

    resolved_age_months = age_months if age_months is not None else (age * 12 if age is not None else None)
    amber_reasons: List[str] = []
    if spo2 is not None and 92 <= spo2 <= 93:
        amber_reasons.append(f"SpO2 {spo2}% is in the primer amber range (92–93%).")
    if temperature_c is not None and resolved_age_months is not None \
            and resolved_age_months >= 3 and temperature_c >= 39.1:
        amber_reasons.append(
            f"Temperature {temperature_c:.1f}°C is at least 39.1°C at age 3 months or older."
        )
    resolved_age_years = (
        age_months / 12 if age_months is not None else age
    )
    if resolved_age_years is not None and resolved_age_years >= 16 \
            and pulse is not None and 111 <= pulse <= 130:
        amber_reasons.append(f"Pulse {pulse} bpm is in the adult amber range (111–130 bpm).")
    l2_rules = HARD_ESCALATION_VITALS["L2"]
    if pregnant is True and bp_systolic is not None \
            and l2_rules["pregnant_sbp_amber_low"] <= bp_systolic < l2_rules["pregnant_sbp_amber_high"]:
        amber_reasons.append(
            f"Pregnant patient SBP {bp_systolic} mmHg is in the 140–159 mmHg amber range."
        )
    if pregnant is True and bp_diastolic is not None \
            and l2_rules["pregnant_dbp_amber_low"] <= bp_diastolic < l2_rules["pregnant_dbp_amber_high"]:
        amber_reasons.append(
            f"Pregnant patient DBP {bp_diastolic} mmHg is in the 90–109 mmHg amber range."
        )
    if amber_reasons and final_tier > 3:
        final_tier = 3
        all_reasons.extend(f"Primer amber vital: {reason} Priority raised to Level 3." for reason in amber_reasons)

    if resolved_age_months is not None and resolved_age_months < 3 and final_tier == 5:
        final_tier = 4
        all_reasons.append(
            "Primer infant rule: patients under 3 months cannot be Level 5; priority raised to Level 4."
        )

    if final_tier in [1, 2]:
        department = "Emergency/Casualty"
    elif resolved_age_years is not None and resolved_age_years < 12:
        department = recommend_department(
            complaint, age, gender, final_tier, age_months, pregnant, recently_delivered
        )
    elif pregnant is True or recently_delivered:
        routed_department = recommend_department(
            complaint, age, gender, final_tier, age_months, pregnant, recently_delivered
        )
        if routed_department == "Obstetrics & Gynaecology":
            department = routed_department

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
    Requires a clinical rationale, keeps emergency tiers in Emergency/Casualty, and
    prevents overrides from downgrading a hard-rule escalation. The confirmation
    parameter is retained for API compatibility but is not clinician authentication.
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

    if new_tier in [1, 2] and new_dept != "Emergency/Casualty":
        return False, "Override rejected: Level 1 and Level 2 patients must be routed to Emergency/Casualty."

    if current_tier in [1, 2] and new_tier is not None and new_tier > 2:
        return False, (
            "Safety Guardrail: A Level 1 or Level 2 encounter cannot be downgraded "
            "to a non-emergency tier through a nurse override. Re-triage with updated "
            "clinical information instead."
        )

    if is_escalated and current_tier in [1, 2] and new_tier is not None and new_tier > current_tier:
        return False, (
            "Safety Guardrail: This patient triggered a hard-coded clinical escalation. "
            "A nurse override cannot downgrade that escalation; perform a documented "
            "clinical re-triage with updated information."
        )

    return True, None
