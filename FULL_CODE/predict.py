"""CarePriority Gov - Batch Prediction Utility (predict.py)
======================================================
Reads an intake records CSV and writes hybrid triage & department predictions:
1. record_id: ID of the record/patient
2. triage_level: 1-5 (Resuscitation to Non-Urgent) or blank ("") if safety abstained
3. department: destination clinical department
4. abstain_flag: 1 if safety abstained, 0 if safely triaged
5. red_flags: semicolon-delimited clinical red flags identified
6. rule_triggered: name of hard-coded escalation/abstention rule, or blank

COLUMN RENAMING CONFIGURATION:
Easily adjust `OUTPUT_COLUMN_MAP` below once organizer evaluate.py expected schema is known.
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from triage_engine import run_triage, clean_vitals_for_storage
from train_model import clean_and_normalize_df

# ==============================================================================
# OUTPUT COLUMN MAPPING CONFIGURATION
# Modify these values to instantly match what organizer evaluate.py expects!
# ==============================================================================
OUTPUT_COLUMN_MAP = {
    "record_id": "record_id",           # e.g. "encounter_id", "id", "patient_id"
    "triage_level": "triage_level",     # e.g. "predicted_tier", "level", "tier", "triage_tier"
    "department": "department",         # e.g. "predicted_department", "target_department", "dept"
    "abstain_flag": "abstain_flag",     # e.g. "abstained", "is_abstained", "abstain"
    "red_flags": "red_flags",           # e.g. "clinical_red_flags", "flags"
    "rule_triggered": "rule_triggered"  # e.g. "escalation_rule", "rule_name", "guardrail_triggered"
}


def predict_records(
    input_csv: str,
    output_csv: str,
    column_mapping: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Executes CarePriority Gov hybrid pipeline on input intake CSV and saves predictions.
    """
    if column_mapping is None:
        column_mapping = OUTPUT_COLUMN_MAP

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    print(f"[CarePriority Predict] Reading intake records from: {input_csv}")
    df_raw = pd.read_csv(input_csv)
    total_records = len(df_raw)
    print(f"[CarePriority Predict] Loaded {total_records} records.")

    # 1. Normalize schema flexibly using shared column mapper
    clean_df, detected_cols = clean_and_normalize_df(df_raw)

    # 2. Extract record identifiers (fallback to index if no ID column)
    id_col = None
    for candidate in ["encounter_id", "record_id", "patient_id", "id", "pt_id"]:
        for col in df_raw.columns:
            if col.lower().strip() == candidate:
                id_col = col
                break
        if id_col:
            break

    if id_col:
        record_ids = df_raw[id_col].astype(str).tolist()
    else:
        record_ids = [f"REC-{idx+1:05d}" for idx in range(total_records)]

    # 3. Batch prediction loop
    pred_rows = []
    abstained_count = 0
    emergency_count = 0

    for idx in range(total_records):
        row = clean_df.iloc[idx]
        rec_id = record_ids[idx]

        complaint = str(row.get("complaint", ""))
        age = int(row["age"]) if pd.notnull(row.get("age")) else None
        gender = str(row.get("sex", "Unknown"))
        spo2 = float(row["spo2"]) if pd.notnull(row.get("spo2")) else None
        bp_sys = int(row["bp_systolic"]) if pd.notnull(row.get("bp_systolic")) else None
        bp_dia = int(row["bp_diastolic"]) if pd.notnull(row.get("bp_diastolic")) else None
        pulse = int(row["pulse"]) if pd.notnull(row.get("pulse")) else None
        temp = float(row["temperature"]) if pd.notnull(row.get("temperature")) else None
        conds = str(row.get("known_conditions", ""))
        meds = str(row.get("medications", ""))

        # Run CarePriority Gov Hybrid Triage Engine
        triage_res = run_triage(
            patient_id=rec_id,
            patient_name=None,
            age=age,
            gender=gender,
            spo2=spo2,
            bp_systolic=bp_sys,
            bp_diastolic=bp_dia,
            pulse=pulse,
            temperature=temp,
            complaint=complaint,
            known_conditions=conds,
            medications=meds
        )

        is_abstained = triage_res["abstained"]
        final_tier = triage_res["tier"]  # None if abstained
        tier_out = str(final_tier) if (final_tier is not None and not is_abstained) else ""
        dept_out = triage_res["department"]
        abstain_flag = 1 if is_abstained else 0

        # Red flags and rules triggered
        red_flags_str = "; ".join(triage_res.get("red_flags", []))
        rule_str = triage_res.get("escalation_rule") or triage_res.get("abstention_reason") or ""

        if is_abstained:
            abstained_count += 1
        elif final_tier in [1, 2]:
            emergency_count += 1

        pred_rows.append({
            column_mapping["record_id"]: rec_id,
            column_mapping["triage_level"]: tier_out,
            column_mapping["department"]: dept_out,
            column_mapping["abstain_flag"]: abstain_flag,
            column_mapping["red_flags"]: red_flags_str,
            column_mapping["rule_triggered"]: rule_str
        })

    df_out = pd.DataFrame(pred_rows)
    df_out.to_csv(output_csv, index=False)
    print(f"[CarePriority Predict] Saved {len(df_out)} predictions to: {output_csv}")
    print(f"[CarePriority Predict] Summary: Total={total_records}, Emergent(L1/L2)={emergency_count}, Abstained={abstained_count} ({abstained_count/max(1, total_records)*100:.1f}%)")

    return df_out


def main():
    parser = argparse.ArgumentParser(
        description="CarePriority Gov - Intake Batch Predictor for Hackathon Evaluation"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=os.path.join("data", "records.csv"),
        help="Path to intake records CSV (default: data/records.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="predictions.csv",
        help="Path to save output predictions CSV (default: predictions.csv)"
    )
    args = parser.parse_args()

    # Fallback to alternative input paths if default not found
    input_path = args.input
    if not os.path.exists(input_path):
        for alt in [
            os.path.join("data", "intake_records.csv"),
            os.path.join("data", "validation_split.csv"),
            os.path.join("data", "synthetic_placeholder.csv")
        ]:
            if os.path.exists(alt):
                print(f"[Notice] '{input_path}' not found. Defaulting to existing dataset: '{alt}'")
                input_path = alt
                break

    predict_records(input_path, args.output)


if __name__ == "__main__":
    main()
