#!/usr/bin/env python3
"""CarePriority Gov organizer dataset adapter, model training, safety layer, and submission generator.

Uses the official synthetic hackathon dataset. This is a clinician-assisting demo only,
not a clinically validated triage system.
"""
from __future__ import annotations
import argparse, json
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
ORGANIZER_ROOT = ROOT / "data" / "organizer"
DATA = ORGANIZER_ROOT / "data"
MODEL_DIR = ROOT / "models" / "organizer"
SUBMISSION = ROOT / "submission"
BASELINE_PATH = ORGANIZER_ROOT / "baseline" / "baseline.py"
safety_baseline = None


def load_safety_baseline():
    """Load the separately supplied organizer baseline only when pipeline work needs it."""
    global safety_baseline
    if safety_baseline is not None:
        return safety_baseline
    if not BASELINE_PATH.is_file():
        raise FileNotFoundError(
            "Organizer baseline is missing. Extract the official organizer package "
            f"under {ORGANIZER_ROOT}; expected {BASELINE_PATH}."
        )

    baseline_spec = importlib.util.spec_from_file_location(
        "organizer_baseline", BASELINE_PATH
    )
    if baseline_spec is None or baseline_spec.loader is None:
        raise ImportError(f"Could not load organizer baseline from {BASELINE_PATH}")
    loaded_baseline = importlib.util.module_from_spec(baseline_spec)
    baseline_spec.loader.exec_module(loaded_baseline)
    safety_baseline = loaded_baseline
    return safety_baseline

NUMERIC = ["age_years", "age_months", "bp_systolic", "bp_diastolic", "pulse_bpm", "temp_c", "spo2_pct"]
CATEGORICAL = ["sex", "pregnant", "arrival_mode"]
TEXT = ["chief_complaint", "known_conditions", "current_medications"]
EXTRA_RF_KEYWORDS = {
    "RF_ECLAMPSIA_SIGNS": ["blurred vision pregnancy", "pregnancy severe headache", "headache in pregnancy", "fit while pregnant", "pregnant with fits"],
    "RF_REDUCED_FETAL_MOVEMENT": ["baby not moving", "reduced fetal movement", "baby movements reduced", "no baby movement"],
    "RF_SEVERE_DEHYDRATION": ["unable to drink", "sunken eyes", "very little urine", "not passing urine", "lethargic with diarrhoea"],
    "RF_CHILD_DANGER_SIGNS": ["child unable to drink", "vomiting everything", "baby not feeding", "child not feeding", "lethargic child"],
    "RF_SEPSIS_SIGNS": ["cold clammy", "cold limbs", "fever with confusion", "chemotherapy with fever"],
    "RF_MENINGITIS_SIGNS": ["neck stiffness with fever", "fever and stiff neck", "fever with neck pain", "child fever confused"],
    "RF_DENGUE_WARNING": ["dengue warning", "fever with bleeding gums", "fever persistent vomiting", "fever with abdominal pain and vomiting"],
    "RF_ACUTE_PSYCHOSIS_AGITATION": ["violent behaviour", "violent behavior", "acutely agitated", "danger to others", "uncontrollable aggression"],
    "RF_DKA_SIGNS": ["diabetic vomiting", "diabetes deep breathing", "high sugar vomiting", "diabetic with abdominal pain and vomiting"],
    "RF_HYPOGLYCAEMIA": ["low sugar", "hypoglycemia", "hypoglycaemia", "diabetic sweating and shaking", "insulin confused"],
    "RF_HEAT_STROKE": ["heat stroke", "heatstroke", "hot body and confusion", "collapse after heat exposure"],
    "RF_ACUTE_ABDOMEN": ["hard abdomen", "rigid abdomen", "distended abdomen", "severe abdominal pain", "painful hernia", "no stool or gas"],
    "RF_TESTICULAR_TORSION": ["testicular pain", "testicle pain", "sudden scrotal pain"],
}
DEPARTMENTS = [
    "General Medicine", "Paediatrics", "Obstetrics & Gynaecology", "General Surgery",
    "Orthopaedics", "ENT", "Ophthalmology", "Dermatology", "Psychiatry", "Dental",
    "Cardiology", "Pulmonology/Chest & TB", "Emergency/Casualty"
]

def clean_vitals_df(df):
    cleaned_input = df.copy()
    for column in ["bp_systolic", "bp_diastolic", "pulse_bpm", "temp_c", "spo2_pct"]:
        cleaned_input[column] = pd.to_numeric(cleaned_input[column], errors="coerce")
    return load_safety_baseline().clean_vitals(cleaned_input)


def feature_frame(df):
    x = df.copy()
    for c in NUMERIC:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    cleaned_vitals = clean_vitals_df(x)
    x["bp_systolic"] = cleaned_vitals["sbp"]
    x["bp_diastolic"] = cleaned_vitals["dbp"]
    x["pulse_bpm"] = cleaned_vitals["hr"]
    x["temp_c"] = cleaned_vitals["temp"]
    x["spo2_pct"] = cleaned_vitals["spo2"]
    for c in CATEGORICAL + TEXT:
        x[c] = x[c].fillna("").astype(str)
    # Timestamp and visit_id intentionally excluded: they are identifiers/time leakage risks.
    return x[NUMERIC + CATEGORICAL + TEXT]

def make_preprocessor():
    transformers = [
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                              ("scale", StandardScaler())]), NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ]
    for col in TEXT:
        transformers.append(("txt_" + col, TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=35000, sublinear_tf=True), col))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)

def fit_models(train_df, labels_df, output_dir: Path | None = MODEL_DIR):
    merged = train_df.merge(labels_df, on="visit_id", how="inner", validate="one_to_one")
    if len(merged) != len(train_df) or len(merged) != len(labels_df):
        raise ValueError("Training intake and label visit_id values do not form a one-to-one join.")
    x = feature_frame(merged)
    triage = Pipeline([("features", make_preprocessor()),
                       ("classifier", LogisticRegression(max_iter=1200, class_weight="balanced", C=1.5))])
    dept = Pipeline([("features", make_preprocessor()),
                     ("classifier", LogisticRegression(max_iter=1200, class_weight="balanced", C=2.0))])
    triage.fit(x, merged["triage_level"].astype(int))
    dept.fit(x, merged["department"].astype(str))
    meta = {
        "training_rows": int(len(merged)),
        "training_visit_ids_unique": bool(merged.visit_id.is_unique),
        "triage_classes": sorted(map(int, triage.classes_)),
        "department_classes": sorted(map(str, dept.classes_)),
        "features": NUMERIC + CATEGORICAL + TEXT,
        "excluded_features": ["visit_id", "arrival_ts"],
        "data_notice": "Synthetic hackathon data only; not for clinical use.",
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(triage, output_dir / "organizer_triage_pipeline.joblib")
        joblib.dump(dept, output_dir / "organizer_department_pipeline.joblib")
        (output_dir / "organizer_model_metadata.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
    return triage, dept, meta

import re

UNINFORMATIVE_EXTENDED = {
    "", "-", "--", ".", "?", "??", "na", "nan", "opd", "same", "check", "xx", "....",
    "dikhana hai", "doctor ko dikhana hai", "doctor", "dikhao", "dikhaane", "doctor ko dikhana", "opd dikhana"
}

def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _text(value):
    return "" if pd.isna(value) else str(value).strip()


def _age_in_years(row):
    months = _number(row.get("age_months"))
    years = _number(row.get("age_years"))
    if months is not None and 0 <= months < 24:
        return months / 12
    return years


def _level1_rule(row, complaint, conditions, meds, red_flags):
    low = complaint.lower()
    sbp = _number(row.get("bp_systolic"))
    spo2 = _number(row.get("spo2_pct"))
    pulse = _number(row.get("pulse_bpm"))
    age = _age_in_years(row)
    age_months = _number(row.get("age_months"))
    if age_months is None and age is not None:
        age_months = age * 12

    if re.search(r"unresponsive|unconscious|not responding|not breathing|gasping|blue lips|turning blue", low):
        return True
    if re.search(r"active seizure|seizing now|convulsing now|fits now|still having (?:a )?fit|still convulsing", low):
        return True
    if re.search(
        r"uncontrolled bleeding|massive bleeding|bleeding (?:will not|won't) stop|"
        r"bleeding that will not stop|khoon.{0,20}(?:band nahi|ruk nahi)",
        low,
    ):
        return True
    if "throat closing" in low:
        return True
    if "collapse" in low or "collapsed" in low:
        if "cold clammy" in low and re.search(r"weak (?:and )?fast pulse|weak pulse|fast pulse", low):
            return True
        if "allerg" in low or "swelling" in low:
            return True
    if "pesticide" in conditions or "insecticide" in conditions or "poison" in low:
        if re.search(r"frothing|heavy secretions|secretions", low):
            return True
    if any(term in conditions or term in meds for term in ("diabet", "insulin", "glimepiride", "sugar tablet")):
        if re.search(r"low sugar|hypogly|confused|drowsy|unusual behaviour|odd behaviour", low):
            return True

    severe_breathing = "RF_RESP_DISTRESS" in red_flags
    if spo2 is not None and (spo2 < 85 or (spo2 < 90 and severe_breathing)):
        return True

    if sbp is not None and sbp > 0 and age is not None:
        if age >= 10 and sbp < 80:
            return True
        if age < 10:
            if age_months is not None and age_months < 1:
                hypotension_limit = 60
            elif age_months is not None and age_months < 12:
                hypotension_limit = 70
            else:
                hypotension_limit = 70 + 2 * age
            if sbp < hypotension_limit:
                return True
    if age is not None and age < 8 and pulse is not None and 0 < pulse < 60:
        return True
    return False


def _primer_vitals_level(row, vitals):
    age = _age_in_years(row)
    if age is None:
        return 9
    age_months = _number(row.get("age_months"))
    if age_months is None:
        age_months = age * 12
    pregnant = _text(row.get("pregnant")).upper() == "Y"
    sbp = _number(vitals.get("sbp"))
    dbp = _number(vitals.get("dbp"))
    pulse = _number(vitals.get("hr"))
    temperature = _number(vitals.get("temp"))
    spo2 = _number(vitals.get("spo2"))

    danger = spo2 is not None and spo2 < 92
    danger |= temperature is not None and temperature <= 35.0
    danger |= age_months < 3 and temperature is not None and temperature < 36.0
    danger |= age_months < 3 and temperature is not None and temperature >= 38.0
    if pulse is not None:
        danger |= (
            (age_months < 3 and pulse > 180)
            or (3 <= age_months < 36 and pulse > 160)
            or (3 <= age < 8 and pulse > 140)
            or (8 <= age < 16 and (pulse > 120 or pulse <= 40))
            or (age >= 16 and (pulse >= 131 or pulse <= 40))
        )
    if sbp is not None:
        danger |= age >= 10 and sbp <= 90
        danger |= age >= 16 and sbp >= 220
        danger |= pregnant and sbp >= 160
    if pregnant and dbp is not None:
        danger |= dbp >= 110
    if danger:
        return 2

    amber = (spo2 is not None and 92 <= spo2 <= 93)
    amber |= temperature is not None and age_months >= 3 and temperature >= 39.1
    amber |= age >= 16 and pulse is not None and 111 <= pulse <= 130
    amber |= pregnant and sbp is not None and 140 <= sbp < 160
    amber |= pregnant and dbp is not None and 90 <= dbp < 110
    return 3 if amber else 9


def detect_red_flags(row, low, conditions, meds, v):
    flags = set()
    age = float(row["age_years"]) if pd.notna(row.get("age_years")) else 18.0
    age_months = float(row["age_months"]) if pd.notna(row.get("age_months")) else (age * 12.0)
    temp = v.temp

    # RF_CHEST_PAIN_CARDIAC (English, Hindi, Telugu, Marathi, Bengali)
    chest_pain = re.search(
        r"chest.*(?:pain|pressure|heavy|heaviness|tight)|"
        r"(?:heavy|heaviness|pressure|tight).*chest|"
        r"seene.*dard|chhati.*dard|chhatit.*dukh|buke.*byatha|nenju.*vali|"
        r"chaati.*noppi|chhati.*noppi|gunde.*noppi|gunde.*baadha|"
        r"सीने में दर्द|छाती",
        low,
    )
    clearly_noncardiac_chest_pain = re.search(
        r"(?:pain|dard).{0,40}(?:when cough|coughing|on deep breath|deep breath|"
        r"khaansne pe|after eating|after meals|with food)|"
        r"(?:cough|coughing|khaansne pe).{0,40}(?:chest|seene|chhati).{0,30}(?:pain|dard)",
        low,
    )
    if chest_pain and not clearly_noncardiac_chest_pain:
        flags.add("RF_CHEST_PAIN_CARDIAC")
    diabetes_risk = bool(
        re.search(r"\b(?:diabet\w*|t[12]dm|dm|madhumeh|sugar)\b", conditions)
    )
    if diabetes_risk or age >= 60:
        uneasy = any(
            term in low
            for term in (
                "ghabrahat", "uneasy", "uneasiness", "restless", "bechain", "घबराहट"
            )
        )
        sweating = any(
            term in low
            for term in ("sweat", "pasina", "chematalu", "పసీనా", "పసిన", "पसीना")
        )
        breathless = any(
            term in low
            for term in ("breathless", "breathing difficulty", "saans phool", "aayasam")
        )
        epigastric = any(
            term in low
            for term in ("gas", "burning in stomach", "stomach burning", "pet mein jalan")
        )
        weak_and_dizzy = any(
            term in low for term in ("weakness", "kamzori", "neerasam", "कमजोरी")
        ) and any(
            term in low for term in ("dizz", "chakkar", "kallu tirugut", "चक्कर")
        )
        if (uneasy and (sweating or breathless or epigastric)) or (
            sweating and (breathless or weak_and_dizzy)
        ):
            flags.add("RF_CHEST_PAIN_CARDIAC")

    # RF_STROKE_SIGNS (avoid bare "one side" which catches one side ear ache)
    if re.search(
        r"face.*deviat|mouth.*tilt|muh.*tedha|munh.*tedha|slurred|paralysis|"
        r"one.*side.*weak|ek.*side.*kamzor|ek.*taraf.*haath|"
        r"aadhe.*shareer.*(?:kaam|chal).*band|ek.*dik.*(?:obosh|numb|weak)|"
        r"haath\s+(?:mein\s+)?(?:jaan nahi|kamzor|sun+n)|"
        r"मुँह टेढ़ा|हाथ.*(?:जान नहीं|कमजोर|सुन्न)|"
        r"आधे शरीर.*(?:काम.*बंद|कमजोर|सुन्न)|एक तरफ.*कमजोरी",
        low,
    ):
        flags.add("RF_STROKE_SIGNS")

    # RF_RESP_DISTRESS
    if re.search(r"cannot.*breathe|gasping|saans.*nahi.*le|saans.*bilkul|saans.*ukhad|pasli.*chal|chest.*indrawing|struggling.*breathe|साँस नहीं ले", low):
        flags.add("RF_RESP_DISTRESS")

    # RF_ALTERED_SENSORIUM (English, Hindi, Tamil, Bengali)
    if re.search(r"unconscious|not.*responding|behosh|hosh.*nahi|hosh.*nei|pehchaan.*nahi|mayakkam|बेहोश", low):
        flags.add("RF_ALTERED_SENSORIUM")

    # RF_SEIZURE (English, Hindi, Tamil, Telugu)
    if re.search(r"\bfits\b|jhatke|\bconvuls|seizure|mirgi|daura|valippu|moorcha|झटके|మూర్ఛ", low):
        flags.add("RF_SEIZURE")

    # RF_SEVERE_BLEEDING
    cough_blood = re.search(
        r"(?:cough|khansi|khaansi).*(?:blood|khoon|खून)|"
        r"(?:blood|khoon|खून).*(?:cough|khansi|khaansi)",
        low,
    )
    minor_sputum_blood = re.search(r"streak|trace|little|small|lakeer", low)
    if (
        re.search(
            r"vomiting.*blood|blood.*vomit|khoon.*ulti|black.*stool|kaali.*tatti|"
            r"khoon.{0,20}(?:band nahi|ruk nahi)|"
            r"(?:bleeding|blood loss).*(?:a lot|lots|massive|uncontrolled|won't stop|will not stop)|"
            r"(?:a lot|lots|massive|uncontrolled).*(?:bleeding|blood)|"
            r"katori.*bhar.*(?:khoon|blood)|(?:khoon|blood).*katori.*bhar",
            low,
        )
        or (cough_blood and not minor_sputum_blood)
    ):
        flags.add("RF_SEVERE_BLEEDING")

    # RF_PREGNANCY_BLEEDING
    if (str(row.get("pregnant", "")).upper() == "Y" or any(k in low for k in ["pregnan", "garbh", "after delivery"])) and any(k in low for k in ["bleeding", "blood", "khoon"]):
        flags.add("RF_PREGNANCY_BLEEDING")

    # RF_POISONING (avoid bare "tablets" which flags routine refills)
    if re.search(
        r"pesticide|insecticide|zeher|zaher|keede.*maarne|"
        r"keet+n+a+s+h+a+k|keetanashak|chuha.*maar.*(?:dawai|dava|medicine)|"
        r"kerosene|mitti.*tel|goliyan.*kha|swallowed.*poison|drank.*poison|"
        r"purugula.*mandu|ज़हर|कीटनाशक",
        low,
    ):
        flags.add("RF_POISONING")

    # RF_SNAKEBITE (handles typo "snnake")
    if re.search(r"s+n+a+k+e|saanp|saap|paamu|paambu|shaap|సాँप|పాము", low):
        flags.add("RF_SNAKEBITE")

    # RF_SCORPION_STING
    if re.search(r"scorpion|bichhu|bichhoo|vinchu|తేలు|బిచ్ఛూ", low):
        flags.add("RF_SCORPION_STING")

    # RF_SUICIDAL
    if re.search(
        r"suicide|end.*life|marne.*ka|marne.*ke|jeena.*nahi|मरने|"
        r"self[- ]?harm|hurt myself|cut (?:my |his |her )?wrist|"
        r"kalai.*kaat|khud.*kaat",
        low,
    ):
        flags.add("RF_SUICIDAL")

    # RF_ANAPHYLAXIS
    if re.search(r"lips.*face.*swollen|throat.*closing|gala.*band", low) or (any(k in low for k in ["allergic", "allergy"]) and any(k in low for k in ["swelling", "breathe", "saans"])):
        flags.add("RF_ANAPHYLAXIS")

    # RF_TRAUMA_MAJOR
    if re.search(r"accident|hit.*by|fell.*from.*roof|fell.*from.*tree|chhat.*se.*gir|ped.*se.*gir|head.*injury|thala.*gaayam", low):
        flags.add("RF_TRAUMA_MAJOR")

    # RF_EYE_EMERGENCY (avoid bare "acid" matching "acidity")
    if re.search(
        r"chuna.*aankh|aankh.*chuna|chemical.*eye|eye.*chemical|"
        r"acid.*eye|eye.*acid|tezaab.*aankh|sudden.*loss.*vision|"
        r"(?:one eye|ek aankh).*?(?:sudden|achanak).*?(?:vision|sight|roshni|dikh)|"
        r"(?:vision|sight|roshni).*?(?:sudden|achanak).*?(?:loss|lost|band|chali gayi)|"
        r"achanak.*(?:one eye|ek aankh).*?(?:vision|sight|roshni|dikh).*?(?:loss|lost|band|chali gayi)|"
        r"एक आंख.*(?:रोशनी|दिखना).*(?:अचानक|बंद|चली गई)|"
        r"एक आँख.*(?:रोशनी|दिखना).*(?:अचानक|बंद|चली गई)|"
        r"अचानक.*एक (?:आंख|आँख).*(?:रोशनी|दिखना)",
        low,
    ):
        flags.add("RF_EYE_EMERGENCY")

    # RF_BURNS_MAJOR
    if re.search(r"burnt|burns|jal.*gay|onek.*pure.*gech|pure.*gech.*onek", low):
        flags.add("RF_BURNS_MAJOR")

    # RF_THUNDERCLAP_HEADACHE
    if re.search(r"worst.*headache|sar.*phat", low):
        flags.add("RF_THUNDERCLAP_HEADACHE")

    # RF_HIGH_FEVER_INFANT (Strictly under 3 months = age_months < 3.0)
    if age_months < 3.0 and pd.notna(temp) and temp >= 38.0:
        flags.add("RF_HIGH_FEVER_INFANT")

    # RF_SEPSIS_SIGNS (fever in cancer chemotherapy or fever with oliguria / cold clammy limbs)
    if "chemo" in conditions or "chemotherapy" in conditions or "cheotherapy" in conditions or "chemo" in low:
        if (pd.notna(temp) and temp >= 37.8) or any(k in low for k in ["fever", "bukhar", "temperature", "jor"]):
            flags.add("RF_SEPSIS_SIGNS")
    if any(k in low for k in ["proshrab kom", "peshab kam", "peshab nahi", "little urine", "cold clammy", "cold limbs"]):
        if (pd.notna(temp) and temp >= 38.0) or any(k in low for k in ["fever", "bukhar", "temperature", "jor"]):
            flags.add("RF_SEPSIS_SIGNS")

    # RF_DENGUE_WARNING (acute fever + warning sign)
    fever_present = any(
        k in low
        for k in (
            "fever", "bukhar", "temperature", "jor", "jwar", "jvaram",
            "taap", "kaaychal", "kaaichal",
        )
    )
    dengue_source_exclusion = re.search(
        r"loose motions?|loose stools?|dast|julab|diarrh\w*|"
        r"burning urine|urine burning|peshab.*(?:jalan|jalta)|"
        r"white discharge|vaginal discharge|safed paani|"
        r"jaundice|peeliya|"
        r"(?:right lower|lower|pelvic).{0,20}(?:abdomen|abdominal|stomach).{0,20}pain|"
        r"(?:abdomen|abdominal|stomach).{0,20}pain.{0,20}(?:right lower|lower|pelvic)",
        low,
    )
    if fever_present and not dengue_source_exclusion:
        dengue_warning_signs = (
            "sab ulti", "ulti ho", "vomiting everything", "vomiting again and again",
            "vomiting repeatedly", "baar baar ulti", "pet dard", "pet mein dard",
            "stomach pain", "abdominal pain", "vayiru vali", "vayitru vali",
            "chakkar", "sar ghoom", "kallu tirugut", "dizz",
            "bleeding gums", "bleeding nose", "nose bleeding", "black stool",
            "nalla motion", "lethargic", "restless",
        )
        if any(k in low for k in dengue_warning_signs):
            flags.add("RF_DENGUE_WARNING")

    # RF_HYPOGLYCAEMIA (diabetic with tremors, shaking, sweating, confusion)
    if "diabet" in conditions or "sugar" in conditions or "insulin" in meds or "glimepiride" in meds:
        if any(k in low for k in ["sugar low", "low sugar", "kaanp", "kampan", "shaking", "trembling", "sweating", "confused", "ghabrahat"]):
            flags.add("RF_HYPOGLYCAEMIA")

    # RF_ECLAMPSIA_SIGNS
    if str(row.get("pregnant", "")).upper() == "Y" or any(k in low for k in ["pregnant", "pregnancy", "delivered"]):
        if any(k in low for k in ["blurred vision", "severe headache", "headache and vision", "fits"]):
            flags.add("RF_ECLAMPSIA_SIGNS")

    # RF_CHILD_DANGER_SIGNS
    if age < 5.0 and any(k in low for k in ["unable to drink", "unable to breastfeed", "vomiting everything", "lethargic child", "not feeding"]):
        flags.add("RF_CHILD_DANGER_SIGNS")

    # RF_SEVERE_DEHYDRATION
    if any(k in low for k in ["unable to drink", "sunken eyes", "very little urine", "lethargic with diarrhoea"]):
        flags.add("RF_SEVERE_DEHYDRATION")

    # RF_ACUTE_ABDOMEN
    if any(k in low for k in ["hard abdomen", "rigid abdomen", "distended abdomen", "stuck hernia", "no stool or gas"]):
        flags.add("RF_ACUTE_ABDOMEN")

    # RF_TESTICULAR_TORSION
    if any(k in low for k in ["testicular pain", "testicle pain", "scrotal pain"]):
        flags.add("RF_TESTICULAR_TORSION")

    for code, keywords in EXTRA_RF_KEYWORDS.items():
        if any(keyword in low for keyword in keywords):
            flags.add(code)

    return flags

def safety_predictions(
    df,
    triage_model,
    dept_model,
    emergency_threshold=1.01,
    model_outputs=None,
):
    """Apply the organizer's mandatory escalation layer before prediction or abstention."""
    x = feature_frame(df)
    if model_outputs is None:
        level_probs = triage_model.predict_proba(x)
        department_probs = dept_model.predict_proba(x)
    else:
        level_probs, department_probs = model_outputs
    model_levels = triage_model.classes_[np.argmax(level_probs, axis=1)].astype(int)
    model_depts = dept_model.classes_[np.argmax(department_probs, axis=1)]
    emergency_classes = [
        index for index, level in enumerate(triage_model.classes_) if int(level) in (1, 2)
    ]
    emergency_probabilities = (
        level_probs[:, emergency_classes].sum(axis=1)
        if emergency_classes
        else np.zeros(len(df), dtype=float)
    )

    cleaned_v = clean_vitals_df(df).reset_index(drop=True)
    out = []

    for i, row in df.reset_index(drop=True).iterrows():
        vid = _text(row.get("visit_id"))
        complaint = _text(row.get("chief_complaint")).lower()
        conditions = _text(row.get("known_conditions")).lower()
        meds = _text(row.get("current_medications")).lower()
        v = cleaned_v.loc[i]
        age = _age_in_years(row)
        flags_set = detect_red_flags(row, complaint, conditions, meds, v)
        flags = ";".join(sorted(flags_set))
        forced_l1 = _level1_rule(row, complaint, conditions, meds, flags_set)
        vital_level = _primer_vitals_level(row, v)

        if forced_l1:
            out.append((vid, 1, "Emergency/Casualty", flags, ""))
            continue
        if flags_set or vital_level == 2:
            out.append((vid, 2, "Emergency/Casualty", flags, ""))
            continue

        usable_vitals = int(v[["sbp", "hr", "temp", "spo2"]].notna().sum())
        uninformative = complaint in UNINFORMATIVE_EXTENDED
        if age is None or (uninformative and usable_vitals <= 1):
            out.append((vid, "ABSTAIN", "", flags, ""))
            continue

        model_level = int(model_levels[i])
        if emergency_probabilities[i] >= emergency_threshold:
            level = 2
        elif vital_level == 3:
            level = min(model_level, 3)
        else:
            level = model_level

        department = "Emergency/Casualty" if level <= 2 else str(model_depts[i])
        if department not in DEPARTMENTS:
            department = "General Medicine"
        if age < 12 and department in {"General Medicine", "Pulmonology/Chest & TB"} and level > 2:
            department = "Paediatrics"

        confidence = (
            ""
            if level != model_level
            else round(float(np.max(level_probs[i])), 4)
        )

        out.append((vid, level, department, flags, confidence))

    return pd.DataFrame(out, columns=["visit_id", "triage_level", "department", "red_flags", "confidence"])


def load_official_evaluator():
    evaluator_path = ORGANIZER_ROOT / "evaluate.py"
    if not evaluator_path.is_file():
        raise FileNotFoundError(f"Official organizer evaluator not found: {evaluator_path}")
    spec = importlib.util.spec_from_file_location("organizer_evaluator", evaluator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load official evaluator from {evaluator_path}")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    return evaluator


def choose_emergency_threshold(calibration_df, calibration_labels, triage_model, dept_model, evaluator):
    features = feature_frame(calibration_df)
    level_probabilities = triage_model.predict_proba(features)
    department_probabilities = dept_model.predict_proba(features)
    emergency_classes = [
        index for index, level in enumerate(triage_model.classes_) if int(level) in (1, 2)
    ]
    if not emergency_classes:
        raise ValueError("The trained triage model has no emergency classes (levels 1 or 2).")

    emergency_probabilities = level_probabilities[:, emergency_classes].sum(axis=1)
    emergency_mask = calibration_labels["true_emergency"].astype(int).to_numpy() == 1
    thresholds = np.unique(
        np.append(
            np.append(
                np.quantile(emergency_probabilities, np.linspace(0, 1, 21)),
                emergency_probabilities[emergency_mask],
            ),
            np.nextafter(float(emergency_probabilities.max()), np.inf),
        )
    )
    best_threshold = None
    best_key = None

    import tempfile
    with tempfile.TemporaryDirectory(prefix="carepriority-calibration-") as temp_dir:
        pred_path = Path(temp_dir) / "predictions.csv"
        truth_path = Path(temp_dir) / "truth.csv"
        calibration_labels.to_csv(truth_path, index=False, encoding="utf-8")
        truth = evaluator.load_truth(str(truth_path))

        for threshold in thresholds:
            predictions = safety_predictions(
                calibration_df,
                triage_model,
                dept_model,
                emergency_threshold=float(threshold),
                model_outputs=(level_probabilities, department_probabilities),
            )
            predictions.to_csv(pred_path, index=False, encoding="utf-8")
            score = evaluator.score(evaluator.load_submission(str(pred_path)), truth)
            key = (
                float(score["primary_score"]),
                float(score["emergency_recall"]),
                float(threshold),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_threshold = float(threshold)

    return best_threshold, {
        "primary_score": best_key[0],
        "emergency_recall": best_key[1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="train models using organizer train files")
    ap.add_argument("--predict", action="store_true", help="generate official-format test submission")
    ap.add_argument("--evaluate-val", action="store_true", help="train on earlier visits and score a later held-out time split")
    ap.add_argument("--evaluate-train", action="store_true", help="quick in-sample diagnostic only (not a generalization score)")
    ap.add_argument("--output", type=Path, default=SUBMISSION / "H1_predictions.csv", help="prediction output path")
    args = ap.parse_args()

    required_paths = [
        BASELINE_PATH,
        DATA / "intake_train.csv",
        DATA / "train_labels.csv",
    ]
    if args.predict:
        required_paths.append(DATA / "intake_test.csv")
    if args.evaluate_val:
        required_paths.append(ORGANIZER_ROOT / "evaluate.py")
    missing_paths = [path for path in required_paths if not path.is_file()]
    if missing_paths:
        missing_list = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(
            "The requested organizer pipeline operation requires the official "
            f"organizer package under {ORGANIZER_ROOT}. Missing files:\n{missing_list}"
        )

    train_df = pd.read_csv(DATA / "intake_train.csv")
    labels_df = pd.read_csv(DATA / "train_labels.csv")
    triage_model = dept_model = None

    if args.evaluate_val:
        merged = train_df.merge(labels_df, on="visit_id", how="inner", validate="one_to_one")
        merged["arrival_ts"] = pd.to_datetime(merged["arrival_ts"], errors="coerce")
        if merged["arrival_ts"].isna().any():
            raise ValueError("Training data contains missing or invalid arrival_ts values.")
        merged = merged.sort_values("arrival_ts", kind="stable").reset_index(drop=True)
        train_end = int(len(merged) * 0.7)
        calibration_end = int(len(merged) * 0.8)
        train_part = merged.iloc[:train_end]
        calibration_part = merged.iloc[train_end:calibration_end]
        validation_part = merged.iloc[calibration_end:]
        sub_train_df = train_part[train_df.columns]
        sub_train_lbl = train_part[labels_df.columns]
        calibration_df = calibration_part[train_df.columns]
        calibration_labels = calibration_part[labels_df.columns]
        sub_val_df = validation_part[train_df.columns]
        sub_val_lbl = validation_part[labels_df.columns]

        print(
            f"Training on {len(sub_train_df)} earliest visits "
            f"({train_part['arrival_ts'].min()} to {train_part['arrival_ts'].max()})..."
        )
        triage_m, dept_m, _ = fit_models(sub_train_df, sub_train_lbl, output_dir=None)
        official_evaluator = load_official_evaluator()
        emergency_threshold, calibration_metrics = choose_emergency_threshold(
            calibration_df,
            calibration_labels,
            triage_m,
            dept_m,
            official_evaluator,
        )
        print(
            f"Selected emergency threshold {emergency_threshold:.6f} "
            f"on {len(calibration_df)} later calibration visits "
            f"(primary score {calibration_metrics['primary_score']:.4f}, "
            f"emergency recall {calibration_metrics['emergency_recall']:.4f})."
        )
        print(
            f"Scoring on {len(sub_val_df)} later held-out visits "
            f"({validation_part['arrival_ts'].min()} to {validation_part['arrival_ts'].max()})..."
        )
        val_preds = safety_predictions(
            sub_val_df,
            triage_m,
            dept_m,
            emergency_threshold=emergency_threshold,
        )

        import tempfile
        with tempfile.TemporaryDirectory(prefix="carepriority-validation-") as temp_dir:
            val_pred_path = Path(temp_dir) / "predictions.csv"
            val_truth_path = Path(temp_dir) / "truth.csv"
            baseline_path = Path(temp_dir) / "baseline.csv"
            val_preds.to_csv(val_pred_path, index=False, encoding="utf-8")
            sub_val_lbl.to_csv(val_truth_path, index=False, encoding="utf-8")
            load_safety_baseline().predict(sub_val_df).to_csv(
                baseline_path, index=False, encoding="utf-8"
            )
            sub = official_evaluator.load_submission(str(val_pred_path))
            truth = official_evaluator.load_truth(str(val_truth_path))
            res = official_evaluator.score(sub, truth)
            baseline_result = official_evaluator.score(
                official_evaluator.load_submission(str(baseline_path)),
                truth,
            )

        res["validation_method"] = (
            "Chronological 70/10/20 split by arrival_ts: earliest 70% for training, "
            "next 10% for emergency-threshold calibration, latest 20% for held-out scoring. "
            "The organizer schema has no patient identifier."
        )
        res["training_rows"] = int(len(sub_train_df))
        res["calibration_rows"] = int(len(calibration_df))
        res["validation_rows"] = int(len(sub_val_df))
        res["emergency_threshold"] = emergency_threshold
        res["calibration_primary_score"] = calibration_metrics["primary_score"]
        res["calibration_emergency_recall"] = calibration_metrics["emergency_recall"]
        res["baseline_primary_score"] = float(baseline_result["primary_score"])
        res["baseline_emergency_recall"] = float(baseline_result["emergency_recall"])
        res["training_visit_id_unique"] = bool(sub_train_df.visit_id.is_unique)
        res["calibration_visit_id_unique"] = bool(calibration_df.visit_id.is_unique)
        res["validation_visit_id_unique"] = bool(sub_val_df.visit_id.is_unique)
        res["validation_start"] = validation_part["arrival_ts"].min().isoformat()
        res["validation_end"] = validation_part["arrival_ts"].max().isoformat()
        official_evaluator.report(res)
        score_path = SUBMISSION / "organizer_validation_score.json"
        SUBMISSION.mkdir(parents=True, exist_ok=True)
        with open(score_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"Validation score report saved to {score_path}")
        return

    if args.train or args.predict or args.evaluate_train:
        triage_model, dept_model, meta = fit_models(train_df, labels_df)
        print(f"Trained on {meta['training_rows']} labeled synthetic records.")

    if args.predict:
        score_path = SUBMISSION / "organizer_validation_score.json"
        if not score_path.is_file():
            raise FileNotFoundError(
                "Run `python hackathon_pipeline.py --evaluate-val` first to calibrate "
                "the emergency threshold on a separate chronological calibration split."
            )
        with score_path.open(encoding="utf-8") as score_file:
            emergency_threshold = float(json.load(score_file)["emergency_threshold"])
        test_df = pd.read_csv(DATA / "intake_test.csv")
        pred = safety_predictions(
            test_df,
            triage_model,
            dept_model,
            emergency_threshold=emergency_threshold,
        )
        path = args.output.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        pred.to_csv(path, index=False, encoding="utf-8")
        print(f"Wrote {len(pred)} predictions to {path}")
        print(pred["triage_level"].astype(str).value_counts().sort_index().to_string())
        ids_match = pred.visit_id.tolist() == test_df.visit_id.tolist()
        print(f"Unique IDs: {pred.visit_id.nunique()} / {len(test_df)}; test IDs in original order: {ids_match}")
        if len(pred) != len(test_df) or not ids_match:
            raise ValueError("Generated predictions do not match the test visits in count and order.")

    if args.evaluate_train:
        pred = safety_predictions(train_df, triage_model, dept_model)
        check = labels_df.merge(pred, on="visit_id", suffixes=("_true", "_pred"), validate="one_to_one")
        nonab = check[check.triage_level_pred.astype(str) != "ABSTAIN"]
        print("IN-SAMPLE DIAGNOSTIC ONLY (not held-out performance)")
        print("Triage accuracy among non-abstained:", round(accuracy_score(nonab.triage_level_true.astype(int), nonab.triage_level_pred.astype(int)), 4) if len(nonab) else "n/a")
        print("Department accuracy among non-abstained:", round(accuracy_score(nonab.department_true, nonab.department_pred), 4) if len(nonab) else "n/a")

if __name__ == "__main__":
    main()
