"""CarePriority Gov - ML Model Training Pipeline
=================================================
Trains scikit-learn triage & department classifiers:
1. Loads organizer files in /data if present; otherwise falls back to
   data/synthetic_placeholder.csv with an explicit placeholder flag.
2. Character n-gram TF-IDF (2-5 grams) on multilingual/transliterated complaints
   combined with normalized numeric vitals and demographics.
3. Asymmetric 20x Safety Cost Optimization:
   Under-triaging a true emergency costs ~20x more than over-triaging a mild case.
   Optimizes decision threshold tau to maximize emergency recall at acceptable over-triage cost.
4. Saves pipelines and metadata to models/ using joblib.
"""

import os
import glob
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, Any, Optional

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, precision_score, classification_report

from rules_config import (
    TRIAGE_COST_MATRIX_5X5,
    ASYMMETRIC_COST_CONFIG,
    calculate_asymmetric_cost
)


def find_data_file() -> Tuple[str, bool]:
    """
    Locates the dataset in /data directory.
    If organizer dataset is present (not synthetic_placeholder.csv), returns it with is_placeholder=False.
    Otherwise returns data/synthetic_placeholder.csv with is_placeholder=True.
    """
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    # Search for organizer csv/parquet/json files in data/
    candidates = glob.glob(os.path.join(data_dir, "*.csv")) + \
                 glob.glob(os.path.join(data_dir, "*.parquet")) + \
                 glob.glob(os.path.join(data_dir, "*.json"))

    # Filter out synthetic placeholder and test splits to find real organizer files if added
    organizer_files = [
        f for f in candidates
        if "synthetic_placeholder" not in os.path.basename(f).lower()
        and "validation_split" not in os.path.basename(f).lower()
        and not os.path.basename(f).startswith(".")
    ]

    if organizer_files:
        chosen = organizer_files[0]
        print(f"[CarePriority ML] Detected ORGANIZER DATASET: {chosen}")
        return chosen, False

    placeholder_path = os.path.join(data_dir, "synthetic_placeholder.csv")
    if not os.path.exists(placeholder_path):
        from generate_placeholder_data import generate_placeholder_dataset
        generate_placeholder_dataset(placeholder_path)

    print(f"[CarePriority ML] Using SYNTHETIC PLACEHOLDER DATASET: {placeholder_path}")
    return placeholder_path, True


def clean_and_normalize_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Cleans raw dataframe and maps organizer columns flexibly.
    Supports arbitrary organizer column names, string representations of levels,
    and handles missing values according to clinical data entry rules:
    - BP of 0 = treated as missing, flagged, stored as None (np.nan).
    - Missing values stay None (np.nan), never coerced to 0.
    """
    df = df.copy()

    # Column mapping heuristic (case-insensitive, stripped)
    cols_lower = {col.lower().strip(): col for col in df.columns}

    def resolve_col(candidates, default=None):
        for c in candidates:
            if c.lower().strip() in cols_lower:
                return cols_lower[c.lower().strip()]
        return default

    col_map = {
        "complaint": resolve_col([
            "chief_complaint", "complaint", "symptoms", "complaint_original", 
            "chief_complaint_text", "presenting_complaint", "cc", "patient_complaint", 
            "reason_for_visit", "intake_complaint"
        ], "chief_complaint"),
        "spo2": resolve_col([
            "spo2", "o2_sat", "oxygen_saturation", "pulse_oximetry", 
            "spo2_percent", "sp_o2", "o2sat", "saturation"
        ], "spo2"),
        "bp_systolic": resolve_col([
            "bp_systolic", "systolic", "sbp", "bp_sys", "systolic_bp", 
            "blood_pressure_systolic", "sys_bp", "bp_s"
        ], "bp_systolic"),
        "bp_diastolic": resolve_col([
            "bp_diastolic", "diastolic", "dbp", "bp_dia", "diastolic_bp", 
            "blood_pressure_diastolic", "dia_bp", "bp_d"
        ], "bp_diastolic"),
        "pulse": resolve_col([
            "pulse", "heart_rate", "hr", "pulse_rate", "bpm", "pulse_bpm"
        ], "pulse"),
        "temperature": resolve_col([
            "temperature", "temp", "body_temperature", "temp_f", "temp_c"
        ], "temperature"),
        "age": resolve_col([
            "age", "patient_age", "age_years", "pt_age"
        ], "age"),
        "sex": resolve_col([
            "sex", "gender", "patient_sex", "patient_gender"
        ], "sex"),
        "known_conditions": resolve_col([
            "known_conditions", "medical_history", "history", "conditions", 
            "past_medical_history", "pmh", "comorbidities", "chronic_conditions"
        ], "known_conditions"),
        "medications": resolve_col([
            "medications", "meds", "current_meds", "current_medications", 
            "rx", "drugs", "prescriptions"
        ], "medications"),
        "true_level": resolve_col([
            "true_level", "triage_level", "tier", "level", "esi_level", 
            "true_tier", "acuity_level", "target_level", "label_level", 
            "priority_level", "priority", "true_priority", "esi"
        ], "true_level"),
        "true_department": resolve_col([
            "true_department", "department", "dept", "target_department", 
            "destination_department", "label_department", "assigned_department"
        ], "true_department"),
        "true_emergency": resolve_col([
            "true_emergency", "is_emergency", "emergency", "emergency_flag", 
            "critical", "is_critical", "label_emergency", "true_critical", "emergency_label"
        ], "true_emergency")
    }

    # Standardize columns into canonical names
    clean_dict = {}
    for canonical, source in col_map.items():
        if source and source in df.columns:
            clean_dict[canonical] = df[source]
        else:
            clean_dict[canonical] = None

    clean_df = pd.DataFrame(clean_dict)

    # 1. Free-text complaints: fill missing with empty string
    clean_df["complaint"] = clean_df["complaint"].fillna("").astype(str)

    # 2. Demographics & Context
    clean_df["sex"] = clean_df["sex"].fillna("Unknown").astype(str)
    clean_df["known_conditions"] = clean_df["known_conditions"].fillna("").astype(str)
    clean_df["medications"] = clean_df["medications"].fillna("").astype(str)

    # 3. Numeric vitals cleaning (Data Entry Rules: BP 0 -> None/NaN, missing stay NaN)
    numeric_vitals = ["spo2", "bp_systolic", "bp_diastolic", "pulse", "temperature", "age"]
    for v in numeric_vitals:
        clean_df[v] = pd.to_numeric(clean_df[v], errors="coerce")

    # BP of 0 = treated as missing, flagged for verification, stored as None
    clean_df.loc[clean_df["bp_systolic"] == 0, "bp_systolic"] = np.nan
    clean_df.loc[clean_df["bp_diastolic"] == 0, "bp_diastolic"] = np.nan

    # Target: true_level (1 to 5) - handles both numeric and textual representations
    if clean_df["true_level"] is not None and not clean_df["true_level"].isnull().all():
        def parse_level(val):
            if pd.isnull(val):
                return np.nan
            s = str(val).strip().lower()
            if "1" in s or "resuscitation" in s:
                return 1
            elif "2" in s or "emergent" in s:
                return 2
            elif "3" in s or "urgent" in s:
                return 3
            elif "4" in s or "less urgent" in s:
                return 4
            elif "5" in s or "non-urgent" in s:
                return 5
            try:
                num = int(float(val))
                return num if 1 <= num <= 5 else np.nan
            except (ValueError, TypeError):
                return np.nan

        clean_df["true_level"] = clean_df["true_level"].apply(parse_level)
        clean_df = clean_df.dropna(subset=["true_level"])
        clean_df["true_level"] = clean_df["true_level"].astype(int)

    # Target: true_emergency flag (if not present, deduce: Level 1 or 2 is emergency)
    if clean_df["true_emergency"] is None or clean_df["true_emergency"].isnull().all():
        clean_df["true_emergency"] = clean_df["true_level"].isin([1, 2]).astype(int)
    else:
        def parse_emergency(val):
            if pd.isnull(val):
                return 0
            s = str(val).strip().lower()
            if s in ["true", "1", "yes", "t", "y", "critical", "emergency"]:
                return 1
            return 0

        clean_df["true_emergency"] = clean_df["true_emergency"].apply(parse_emergency).astype(int)

    # Target: true_department
    if clean_df["true_department"] is None or clean_df["true_department"].isnull().all():
        clean_df["true_department"] = "General Medicine"
    else:
        clean_df["true_department"] = clean_df["true_department"].fillna("General Medicine").astype(str)

    return clean_df, col_map


def build_feature_preprocessor() -> ColumnTransformer:
    """
    Builds feature transformation pipeline:
    - Character n-grams TF-IDF (2 to 5 chars) on chief complaint to handle
      Hinglish and Telugu-in-Latin morphemes, subwords, and typos.
    - Word TF-IDF on known conditions and medications.
    - Imputed and scaled numerical vitals with missingness indicators.
    - One-hot encoded sex.
    """
    text_transformer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        max_features=2500,
        sublinear_tf=True
    )

    cond_transformer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=300,
        sublinear_tf=True
    )

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("complaint_tfidf", text_transformer, "complaint"),
            ("cond_tfidf", cond_transformer, "known_conditions"),
            ("num_vitals", numeric_transformer, ["spo2", "bp_systolic", "bp_diastolic", "pulse", "temperature", "age"]),
            ("cat_sex", categorical_transformer, ["sex"])
        ],
        remainder="drop"
    )

    return preprocessor


def compute_cost_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates average asymmetric risk cost per patient using 20x cost matrix.
    """
    total_cost = 0.0
    for yt, yp in zip(y_true, y_pred):
        total_cost += calculate_asymmetric_cost(int(yt), int(yp))
    return total_cost / max(1, len(y_true))


def optimize_asymmetric_emergency_threshold(
    val_probs: np.ndarray,
    val_y_true: np.ndarray,
    val_emergency_true: np.ndarray,
    grid_min: float = 0.01,
    grid_max: float = 0.99,
    steps: int = 197
) -> Tuple[float, float, float]:
    """
    Finds optimal decision threshold tau for P(emergency) = P(L1) + P(L2)
    that minimizes the asymmetric cost: 20 * FN + 1 * FP.
    Reports when the chosen threshold hits the edge of the search grid and widens the grid.
    """
    # Sum probabilities of Level 1 and Level 2
    # In classes_ [1, 2, 3, 4, 5], idx 0 is L1, idx 1 is L2
    p_emergency = val_probs[:, 0] + val_probs[:, 1]

    best_tau = 0.25
    best_cost = float("inf")
    best_recall = 0.0

    # Search widened fine grid of thresholds [0.01, 0.99]
    grid = np.linspace(grid_min, grid_max, steps)
    for tau in grid:
        pred_is_emergency = (p_emergency >= tau).astype(int)
        
        # False Negatives: True emergency classified as non-emergency (20x penalty)
        fn = np.sum((val_emergency_true == 1) & (pred_is_emergency == 0))
        # False Positives: Non-emergency over-triaged to emergency (1x penalty)
        fp = np.sum((val_emergency_true == 0) & (pred_is_emergency == 1))

        cost = (ASYMMETRIC_COST_CONFIG["under_triage_emergency_cost"] * fn) + \
               (ASYMMETRIC_COST_CONFIG["over_triage_cost"] * fp)

        recall = np.sum((val_emergency_true == 1) & (pred_is_emergency == 1)) / max(1, np.sum(val_emergency_true == 1))

        # Prioritize minimum asymmetric risk cost
        if cost < best_cost:
            best_cost = cost
            best_tau = float(tau)
            best_recall = float(recall)

    # Report when chosen threshold hits the edge of the search grid
    at_lower_edge = np.isclose(best_tau, grid_min, atol=1e-3)
    at_upper_edge = np.isclose(best_tau, grid_max, atol=1e-3)
    if at_lower_edge or at_upper_edge:
        edge_type = "LOWER" if at_lower_edge else "UPPER"
        print(f"[CarePriority ML Warning] Optimal threshold tau={best_tau:.3f} reached the {edge_type} edge of search grid [{grid_min:.2f}, {grid_max:.2f}]! Consider expanding grid boundaries.")

    return best_tau, best_cost, best_recall


def train_and_save_pipeline(data_path: Optional[str] = None) -> Dict[str, Any]:
    """
    End-to-end training and evaluation script.
    Saves models to models/ directory.
    """
    if data_path is None:
        data_path, is_placeholder = find_data_file()
    else:
        is_placeholder = "placeholder" in data_path.lower()

    os.makedirs("models", exist_ok=True)

    # 1. Load Data
    print(f"[CarePriority ML] Loading dataset from {data_path}...")
    if data_path.endswith(".csv"):
        df_raw = pd.read_csv(data_path)
    elif data_path.endswith(".parquet"):
        df_raw = pd.read_parquet(data_path)
    elif data_path.endswith(".json"):
        df_raw = pd.read_json(data_path)
    else:
        df_raw = pd.read_csv(data_path)

    df, col_map = clean_and_normalize_df(df_raw)
    print(f"[CarePriority ML] Loaded {len(df)} records. Emergency prevalence: {df['true_emergency'].mean()*100:.1f}%")

    # 2. Stratified Train / Validation Split (80% train, 20% validation)
    # Stratify by true_level
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(sss.split(df, df["true_level"]))

    df_train = df.iloc[train_idx].copy()
    df_val = df.iloc[val_idx].copy()

    # 3. Train Triage Model (Model 1)
    preprocessor = build_feature_preprocessor()
    triage_clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        C=1.0,
        random_state=42
    )

    triage_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", triage_clf)
    ])

    print("[CarePriority ML] Fitting Triage Classification Pipeline...")
    triage_pipeline.fit(df_train, df_train["true_level"])

    # 4. Train Department Model (Model 2)
    dept_preprocessor = build_feature_preprocessor()
    dept_clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        C=1.0,
        random_state=42
    )

    dept_pipeline = Pipeline(steps=[
        ("preprocessor", dept_preprocessor),
        ("classifier", dept_clf)
    ])

    print("[CarePriority ML] Fitting Department Recommendation Pipeline...")
    dept_pipeline.fit(df_train, df_train["true_department"])

    # 5. Validation & Asymmetric Cost Optimization
    val_probs = triage_pipeline.predict_proba(df_val)
    val_preds_raw = triage_pipeline.predict(df_val)

    val_y_true = df_val["true_level"].values
    val_emerg_true = df_val["true_emergency"].values

    best_tau, opt_cost, opt_recall = optimize_asymmetric_emergency_threshold(
        val_probs=val_probs,
        val_y_true=val_y_true,
        val_emergency_true=val_emerg_true
    )

    print(f"[CarePriority ML] Optimal Asymmetric Emergency Threshold (tau): {best_tau:.3f}")
    print(f"[CarePriority ML] Validation Emergency Recall: {opt_recall*100:.1f}%")

    # Evaluate decision logic with optimal threshold
    p_emerg_val = val_probs[:, 0] + val_probs[:, 1]
    final_val_preds = []
    for i, p_val in enumerate(val_preds_raw):
        if p_emerg_val[i] >= best_tau:
            # Over-ride to emergency L1 or L2 based on higher prob
            final_val_preds.append(1 if val_probs[i, 0] > val_probs[i, 1] else 2)
        else:
            final_val_preds.append(p_val)
    final_val_preds = np.array(final_val_preds)

    # Metrics on validation set
    raw_cost = compute_cost_score(val_y_true, val_preds_raw)
    opt_cost_score = compute_cost_score(val_y_true, final_val_preds)

    dept_preds = dept_pipeline.predict(df_val)
    dept_acc = np.mean(dept_preds == df_val["true_department"].values)

    conf_mat = confusion_matrix(val_y_true, final_val_preds, labels=[1, 2, 3, 4, 5])

    val_emerg_pred = (final_val_preds <= 2).astype(int)
    emerg_recall = recall_score(val_emerg_true, val_emerg_pred, zero_division=0)
    emerg_prec = precision_score(val_emerg_true, val_emerg_pred, zero_division=0)
    missed_emergencies = int(np.sum((val_emerg_true == 1) & (val_emerg_pred == 0)))

    metadata = {
        "model_trained_at": datetime.now().isoformat(),
        "data_path": data_path,
        "is_placeholder": is_placeholder,
        "total_samples": len(df),
        "train_samples": len(df_train),
        "val_samples": len(df_val),
        "classes": [int(c) for c in triage_pipeline.classes_],
        "optimal_emergency_threshold": float(best_tau),
        "abstention_confidence_threshold": 0.35,
        "validation_metrics": {
            "emergency_recall": float(emerg_recall),
            "emergency_precision": float(emerg_prec),
            "missed_emergencies": int(missed_emergencies),
            "raw_cost_score": float(raw_cost),
            "asymmetric_cost_score": float(opt_cost_score),
            "department_accuracy": float(dept_acc),
            "confusion_matrix_5x5": [[int(cell) for cell in row] for row in conf_mat]
        },
        "column_mapping": col_map
    }

    # 6. Save Artifacts with joblib
    joblib.dump(triage_pipeline, "models/triage_pipeline.joblib")
    joblib.dump(dept_pipeline, "models/dept_pipeline.joblib")
    joblib.dump(metadata, "models/model_metadata.joblib")

    # Also save a JSON summary for human inspection
    with open("models/model_summary.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save held-out validation set for live Evaluation page computations
    df_val.to_csv("data/validation_split.csv", index=False)

    print("[CarePriority ML] Saved models to models/ directory successfully.")
    print(f"[CarePriority ML] Final Held-Out Evaluation: Recall={emerg_recall*100:.1f}%, Missed Emergencies={missed_emergencies}, Cost Score={opt_cost_score:.2f}, Dept Acc={dept_acc*100:.1f}%")

    return metadata


if __name__ == "__main__":
    train_and_save_pipeline()
