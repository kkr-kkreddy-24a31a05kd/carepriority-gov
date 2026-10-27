#!/usr/bin/env python3
"""CarePriority Gov organizer dataset adapter, model training, safety layer, and submission generator.

Uses the official synthetic hackathon dataset. This is a clinician-assisting demo only,
not a clinically validated triage system.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODEL_DIR = ROOT / "models"
SUBMISSION = ROOT / "submission"
sys.path.insert(0, str(ROOT / "baseline"))
import organizer_baseline as safety_baseline

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

def feature_frame(df):
    x = df.copy()
    for c in NUMERIC:
        x[c] = pd.to_numeric(x[c], errors="coerce")
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

def fit_models(train_df, labels_df, output_dir=MODEL_DIR):
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
    output_dir.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(triage, output_dir / "organizer_triage_pipeline.joblib")
    joblib.dump(dept, output_dir / "organizer_department_pipeline.joblib")
    meta = {
        "training_rows": int(len(merged)),
        "training_visit_ids_unique": bool(merged.visit_id.is_unique),
        "triage_classes": sorted(map(int, triage.classes_)),
        "department_classes": sorted(map(str, dept.classes_)),
        "features": NUMERIC + CATEGORICAL + TEXT,
        "excluded_features": ["visit_id", "arrival_ts"],
        "data_notice": "Synthetic hackathon data only; not for clinical use.",
    }
    (output_dir / "organizer_model_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return triage, dept, meta

import re

UNINFORMATIVE_EXTENDED = {
    "", "-", "--", ".", "?", "??", "na", "nan", "opd", "same", "check", "xx", "....",
    "dikhana hai", "doctor ko dikhana hai", "doctor", "dikhao", "dikhaane", "doctor ko dikhana", "opd dikhana"
}

def clean_vitals_df(df):
    out = pd.DataFrame(index=df.index)
    s = pd.to_numeric(df["bp_systolic"], errors="coerce")
    d = pd.to_numeric(df["bp_diastolic"], errors="coerce")
    swap = s < d
    s, d = np.where(swap, d, s), np.where(swap, s, d)
    out["sbp"] = pd.Series(s, index=df.index).where((s >= 50) & (s <= 260))
    out["dbp"] = pd.Series(d, index=df.index).where((d >= 20) & (d <= 160))
    p = pd.to_numeric(df["pulse_bpm"], errors="coerce")
    out["hr"] = p.where((p >= 25) & (p <= 250))
    t = pd.to_numeric(df["temp_c"], errors="coerce")
    t = np.where((t >= 86) & (t <= 113), (t - 32) * 5 / 9, t)
    out["temp"] = pd.Series(t, index=df.index).where(lambda x: (x >= 30) & (x <= 45))
    o = pd.to_numeric(df["spo2_pct"], errors="coerce")
    out["spo2"] = o.where((o >= 50) & (o <= 100))
    return out

def detect_red_flags(row, low, conditions, meds, v):
    flags = set()
    age = float(row["age_years"]) if pd.notna(row.get("age_years")) else 18.0
    age_months = float(row["age_months"]) if pd.notna(row.get("age_months")) else (age * 12.0)
    temp = v.temp

    # RF_CHEST_PAIN_CARDIAC (English, Hindi, Telugu, Marathi, Bengali)
    if re.search(r"chest.*pain|seene.*dard|chhati.*dard|chhatit.*dukh|buke.*byatha|nenju.*vali|chaati.*noppi|chhati.*noppi|gunde.*noppi|gunde.*baadha|chest.*heaviness|pressure.*chest|सीने में दर्द|छाती", low):
        flags.add("RF_CHEST_PAIN_CARDIAC")
    if ("diabet" in conditions or "sugar" in conditions or age >= 60) and any(k in low for k in ["ghabrahat", "sweating and gas", "sudden uneasiness"]):
        flags.add("RF_CHEST_PAIN_CARDIAC")

    # RF_STROKE_SIGNS (avoid bare "one side" which catches one side ear ache)
    if re.search(r"face.*deviat|mouth.*tilt|muh.*tedha|munh.*tedha|slurred|paralysis|one.*side.*weak|ek.*side.*kamzor|ek.*taraf.*haath|mukham.*vankara|mata.*tadabad|मुँह टेढ़ा", low):
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
    if re.search(r"vomiting.*blood|blood.*vomit|khoon.*ulti|black.*stool|kaali.*tatti|khoon.*band.*nahi|bleeding.*lot|lot.*blood", low):
        flags.add("RF_SEVERE_BLEEDING")

    # RF_PREGNANCY_BLEEDING
    if (str(row.get("pregnant", "")).upper() == "Y" or any(k in low for k in ["pregnan", "garbh", "after delivery"])) and any(k in low for k in ["bleeding", "blood", "khoon"]):
        flags.add("RF_PREGNANCY_BLEEDING")

    # RF_POISONING (avoid bare "tablets" which flags routine refills)
    if re.search(r"pesticide|insecticide|zeher|zaher|keede.*maarne|keetnashak|kerosene|mitti.*tel|goliyan.*kha|swallowed.*poison|drank.*poison|purugula.*mandu|ज़हर|कीटनाशक", low):
        flags.add("RF_POISONING")

    # RF_SNAKEBITE (handles typo "snnake")
    if re.search(r"s+n+a+k+e|saanp|saap|paamu|paambu|shaap|సాँप|పాము", low):
        flags.add("RF_SNAKEBITE")

    # RF_SCORPION_STING
    if re.search(r"scorpion|bichhu|bichhoo|vinchu|తేలు|బిచ్ఛూ", low):
        flags.add("RF_SCORPION_STING")

    # RF_SUICIDAL
    if re.search(r"suicide|end.*life|marne.*ka|marne.*ke|jeena.*nahi|मरने", low):
        flags.add("RF_SUICIDAL")

    # RF_ANAPHYLAXIS
    if re.search(r"lips.*face.*swollen|throat.*closing|gala.*band", low) or (any(k in low for k in ["allergic", "allergy"]) and any(k in low for k in ["swelling", "breathe", "saans"])):
        flags.add("RF_ANAPHYLAXIS")

    # RF_TRAUMA_MAJOR
    if re.search(r"accident|hit.*by|fell.*from.*roof|fell.*from.*tree|chhat.*se.*gir|ped.*se.*gir|head.*injury|thala.*gaayam", low):
        flags.add("RF_TRAUMA_MAJOR")

    # RF_EYE_EMERGENCY (avoid bare "acid" matching "acidity")
    if re.search(r"chuna.*aankh|aankh.*chuna|chemical.*eye|eye.*chemical|acid.*eye|eye.*acid|tezaab.*aankh|sudden.*loss.*vision", low):
        flags.add("RF_EYE_EMERGENCY")

    # RF_BURNS_MAJOR
    if re.search(r"burnt|burns|jal.*gay", low):
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
    if any(k in low for k in ["fever", "bukhar", "temperature", "jor", "jwar"]):
        if any(k in low for k in ["sab ulti", "ulti ho", "vomiting everything", "pet dard", "stomach pain", "chakkar"]):
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

    return flags

def safety_predictions(df, triage_model, dept_model):
    """ML output with non-downgradable baseline red-flag/vital escalation and nurse abstention."""
    x = feature_frame(df)
    level_probs = triage_model.predict_proba(x)
    model_levels = triage_model.classes_[np.argmax(level_probs, axis=1)].astype(int)
    dept_probs = dept_model.predict_proba(x)
    model_depts = dept_model.classes_[np.argmax(dept_probs, axis=1)]

    cleaned_v = clean_vitals_df(df).reset_index(drop=True)
    out = []

    for i, row in df.reset_index(drop=True).iterrows():
        vid = str(row["visit_id"])
        complaint = str(row.get("chief_complaint", "") or "").strip().lower()
        conditions = str(row.get("known_conditions", "") or "").lower()
        meds = str(row.get("current_medications", "") or "").lower()

        v = cleaned_v.loc[i]
        age = float(row["age_years"]) if pd.notna(row.get("age_years")) else 18.0
        age_months = float(row["age_months"]) if pd.notna(row.get("age_months")) else (age * 12.0)
        age_effective = age_months / 12.0 if pd.notna(row.get("age_months")) else age
        preg = str(row.get("pregnant", "") or "")

        # 1. Uninformative complaint check: blank or meaningless -> ABSTAIN for nurse review
        if complaint in UNINFORMATIVE_EXTENDED:
            out.append((vid, "ABSTAIN", "", "", 0.0))
            continue

        # 2. Vital signs danger zone & amber rules
        vl = safety_baseline.vitals_level(age_effective, preg, v)

        # 3. Red flags detection
        flags_set = detect_red_flags(row, complaint, conditions, meds, v)
        flags = ";".join(sorted(flags_set))

        if vl == 1:
            level = 1
        elif flags_set or vl == 2:
            level = 2
        else:
            model_lvl = int(model_levels[i])
            if vl == 3:
                level = min(model_lvl, 3)
            else:
                level = model_lvl

        # Department recommendation:
        # Every Level 1 and Level 2 patient goes to Emergency/Casualty per official primer
        department = "Emergency/Casualty" if level <= 2 else str(model_depts[i])
        if department not in DEPARTMENTS:
            department = "General Medicine"
        # Paediatrics routing for younger children with medical complaints per official primer
        if age < 12 and department in {"General Medicine", "Pulmonology/Chest & TB"} and level > 2:
            department = "Paediatrics"

        confidence = float(np.max(level_probs[i]))
        if level != int(model_levels[i]):
            confidence = min(confidence, 0.75)

        out.append((vid, level, department, flags, round(float(confidence), 4)))

    return pd.DataFrame(out, columns=["visit_id", "triage_level", "department", "red_flags", "confidence"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="train models using organizer train files")
    ap.add_argument("--predict", action="store_true", help="generate official-format test submission")
    ap.add_argument("--evaluate-val", action="store_true", help="train on 80% train split and evaluate on 20% held-out validation split with official evaluator")
    ap.add_argument("--evaluate-train", action="store_true", help="quick in-sample diagnostic only (not a generalization score)")
    args = ap.parse_args()

    train_df = pd.read_csv(DATA / "intake_train.csv")
    labels_df = pd.read_csv(DATA / "train_labels.csv")
    triage_model = dept_model = None

    if args.evaluate_val:
        from sklearn.model_selection import StratifiedShuffleSplit
        sys.path.insert(0, str(DATA))
        import evaluate as official_evaluator
        merged = train_df.merge(labels_df, on="visit_id", how="inner", validate="one_to_one")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        train_idx, val_idx = next(sss.split(merged, merged["triage_level"]))

        sub_train_df = merged.iloc[train_idx][train_df.columns]
        sub_train_lbl = merged.iloc[train_idx][labels_df.columns]
        sub_val_df = merged.iloc[val_idx][train_df.columns]
        sub_val_lbl = merged.iloc[val_idx][labels_df.columns]

        print(f"Training on 80% subset ({len(sub_train_df)} rows)...")
        triage_m, dept_m, _ = fit_models(sub_train_df, sub_train_lbl)
        print(f"Scoring on 20% held-out validation subset ({len(sub_val_df)} rows)...")
        val_preds = safety_predictions(sub_val_df, triage_m, dept_m)

        SUBMISSION.mkdir(parents=True, exist_ok=True)
        val_pred_path = SUBMISSION / "val_predictions.csv"
        val_truth_path = SUBMISSION / "val_truth.csv"
        val_preds.to_csv(val_pred_path, index=False, encoding="utf-8")
        sub_val_lbl.to_csv(val_truth_path, index=False, encoding="utf-8")

        sub = official_evaluator.load_submission(str(val_pred_path))
        truth = official_evaluator.load_truth(str(val_truth_path))
        res = official_evaluator.score(sub, truth)
        official_evaluator.report(res)
        with open(SUBMISSION / "val_evaluation_score.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"Validation score report saved to {SUBMISSION / 'val_evaluation_score.json'}")
        return

    if args.train or args.predict or args.evaluate_train:
        triage_model, dept_model, meta = fit_models(train_df, labels_df)
        print(f"Trained on {meta['training_rows']} labeled synthetic records.")

    if args.predict:
        test_df = pd.read_csv(DATA / "intake_test.csv")
        pred = safety_predictions(test_df, triage_model, dept_model)
        SUBMISSION.mkdir(parents=True, exist_ok=True)
        path = SUBMISSION / "H1_predictions.csv"
        pred.to_csv(path, index=False, encoding="utf-8")
        print(f"Wrote {len(pred)} predictions to {path.relative_to(ROOT)}")
        print(pred["triage_level"].astype(str).value_counts().sort_index().to_string())
        print(f"Unique IDs: {pred.visit_id.nunique()} / {len(test_df)}; all test IDs covered: {set(pred.visit_id)==set(test_df.visit_id)}")

    if args.evaluate_train:
        pred = safety_predictions(train_df, triage_model, dept_model)
        check = labels_df.merge(pred, on="visit_id", suffixes=("_true", "_pred"), validate="one_to_one")
        nonab = check[check.triage_level_pred.astype(str) != "ABSTAIN"]
        print("IN-SAMPLE DIAGNOSTIC ONLY (not held-out performance)")
        print("Triage accuracy among non-abstained:", round(accuracy_score(nonab.triage_level_true.astype(int), nonab.triage_level_pred.astype(int)), 4) if len(nonab) else "n/a")
        print("Department accuracy among non-abstained:", round(accuracy_score(nonab.department_true, nonab.department_pred), 4) if len(nonab) else "n/a")

if __name__ == "__main__":
    main()
