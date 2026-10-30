"""Organizer-data status checks, evaluator invocation, and queue replay utilities."""

import os
import subprocess
import heapq
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional

import rules_config
from rules_config import (
    TRIAGE_COST_MATRIX_5X5,
    ASYMMETRIC_COST_CONFIG,
    calculate_asymmetric_cost,
    TRIAGE_LEVELS
)
from triage_engine import (
    run_triage,
    evaluate_hard_escalations,
    score_tier_without_hard_escalation,
    recommend_department,
    predict_ml_triage,
    clean_vitals_for_storage
)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
ORGANIZER_ROOT = DATA_DIR / "organizer"
ORGANIZER_DATA_DIR = ORGANIZER_ROOT / "data"


def check_organizer_data_status() -> Dict[str, Any]:
    """
    Checks whether official organizer files are present in /data/.
    Returns status and exact user download/placement guidance if missing.
    Gracefully handles absence of the data/ folder without crashing.
    """
    has_data_dir = DATA_DIR.is_dir()
    required_data_files = [
        ORGANIZER_DATA_DIR / "intake_train.csv",
        ORGANIZER_DATA_DIR / "train_labels.csv",
        ORGANIZER_DATA_DIR / "intake_test.csv",
    ]
    organizer_data_file = next((str(path) for path in required_data_files if path.is_file()), None)
    has_organizer_data = all(path.is_file() for path in required_data_files)
    evaluate_candidates = [ORGANIZER_ROOT / "evaluate.py", DATA_DIR / "evaluate.py"]
    evaluate_script_path = next((path for path in evaluate_candidates if path.is_file()), None)
    has_evaluate_script = evaluate_script_path is not None

    missing_instructions = []
    if not has_data_dir:
        missing_instructions.append(
            "1. The project `data/` directory does not exist. Extract the organizer package into `data/organizer/`."
        )
    elif not has_organizer_data:
        missing_instructions.append(
            "1. Ensure `data/organizer/data/intake_train.csv`, `train_labels.csv`, and `intake_test.csv` are present."
        )
    if not has_evaluate_script:
        missing_instructions.append(
            "2. Place the official organizer `evaluate.py` in `data/organizer/`."
        )

    return {
        "has_data_dir": has_data_dir,
        "has_organizer_data": has_organizer_data,
        "organizer_data_file": organizer_data_file,
        "has_evaluate_script": has_evaluate_script,
        "evaluate_script_path": str(evaluate_script_path) if evaluate_script_path else None,
        "is_placeholder": not has_organizer_data,
        "missing_instructions": missing_instructions
    }


def execute_organizer_evaluate_script() -> Optional[str]:
    """
    Runs the preserved organizer evaluator only when both evaluation files exist.
    """
    status = check_organizer_data_status()
    script_path = Path(status["evaluate_script_path"]) if status["evaluate_script_path"] else None
    if script_path is None:
        return None

    import sys
    python_exe = sys.executable or "python"
    validation_pairs = [
        (DATA_DIR / "val_predictions.csv", DATA_DIR / "val_truth.csv"),
        (PROJECT_DIR / "submission" / "val_predictions.csv", PROJECT_DIR / "submission" / "val_truth.csv"),
    ]
    selected_pair = next(
        ((predictions, truth) for predictions, truth in validation_pairs if predictions.is_file() and truth.is_file()),
        None,
    )
    if selected_pair is None:
        return None

    predictions, truth = selected_pair
    cmd = [
        python_exe,
        str(script_path),
        "--submission",
        str(predictions),
        "--truth",
        str(truth),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=PROJECT_DIR
        )
        output = proc.stdout or proc.stderr
        if proc.returncode != 0:
            return f"Official evaluator exited with status {proc.returncode}:\n{output}"
        return output
    except (OSError, subprocess.SubprocessError) as e:
        return f"Execution of the official evaluator failed: {e}"


def get_held_out_validation_df() -> Tuple[pd.DataFrame, bool]:
    """
    Returns held-out validation dataframe for live evaluation.
    """
    val_split_path = "data/validation_split.csv"
    if os.path.exists(val_split_path):
        df_val = pd.read_csv(val_split_path)
        is_placeholder = False
        return df_val, is_placeholder

    placeholder_path = "data/synthetic_placeholder.csv"
    if os.path.exists(placeholder_path):
        df_val = pd.read_csv(placeholder_path)
        return df_val.tail(int(len(df_val) * 0.25)), True

    from generate_placeholder_data import generate_placeholder_dataset
    df = generate_placeholder_dataset(placeholder_path)
    return df.tail(int(len(df) * 0.25)), True


def run_comprehensive_evaluation(force_recompute: bool = False) -> Dict[str, Any]:
    """
    Legacy app-engine diagnostic across 3 paradigms:
    1. Rules Only
    2. Model Only
    3. Hybrid CarePriority Gov Pipeline (Model + Rules + Abstention)

    This diagnostic may use placeholder data or a historical cache. It is not the
    official organizer benchmark and must not be used to claim organizer results.
    """
    import json
    cache_path = os.path.join("data", "cached_evaluation.json")
    if not force_recompute and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["comparison_table"] = pd.DataFrame(data["comparison_table"])
            data["organizer_status"] = check_organizer_data_status()
            data["organizer_evaluate_output"] = execute_organizer_evaluate_script()
            data["is_placeholder"] = bool(data.get("is_placeholder", False))
            data["is_cached"] = True
            return data
        except Exception:
            pass

    df_val, is_placeholder = get_held_out_validation_df()
    total_samples = len(df_val)

    # Initialize tracking structures for 3 paradigms
    paradigms = ["rules_only", "model_only", "hybrid"]
    records = {p: [] for p in paradigms}

    for _, row in df_val.iterrows():
        enc_id = str(row.get("encounter_id", "VAL"))
        complaint = str(row.get("complaint", row.get("chief_complaint", "")))
        age = int(row["age"]) if pd.notnull(row.get("age")) else None
        age_months = float(row["age_months"]) if pd.notnull(row.get("age_months")) else None
        pregnancy_value = row.get("pregnant")
        if pd.isnull(pregnancy_value):
            pregnant = None
        else:
            pregnancy_value = str(pregnancy_value).strip().upper()
            pregnant = (
                True if pregnancy_value in {"1", "TRUE", "YES", "Y"}
                else False if pregnancy_value in {"0", "FALSE", "NO", "N"}
                else None
            )
        pregnancy_stage_value = row.get("pregnancy_stage")
        pregnancy_stage = (
            str(pregnancy_stage_value) if pd.notnull(pregnancy_stage_value) else None
        )
        delivery_value = str(row.get("recently_delivered", "")).strip().lower()
        recently_delivered = delivery_value in {"1", "true", "yes", "y"}
        gender = str(row.get("sex", row.get("gender", "Unknown")))
        spo2 = float(row["spo2"]) if pd.notnull(row.get("spo2")) else None
        bp_sys = int(row["bp_systolic"]) if pd.notnull(row.get("bp_systolic")) else None
        bp_dia = int(row["bp_diastolic"]) if pd.notnull(row.get("bp_diastolic")) else None
        pulse = int(row["pulse"]) if pd.notnull(row.get("pulse")) else None
        temp = float(row["temperature"]) if pd.notnull(row.get("temperature")) else None
        conds = str(row.get("known_conditions", "")) if pd.notnull(row.get("known_conditions")) else ""
        meds = str(row.get("medications", "")) if pd.notnull(row.get("medications")) else ""

        gt_tier = int(row["true_level"]) if pd.notnull(row.get("true_level")) else 3
        gt_emerg = int(row["true_emergency"]) if pd.notnull(row.get("true_emergency")) else (1 if gt_tier in [1, 2] else 0)
        gt_dept = str(row.get("true_department", "General Medicine"))

        # Clean vitals (BP of 0 treated as missing)
        clean_v = clean_vitals_for_storage(bp_sys, bp_dia, spo2, pulse, temp)
        sbp = clean_v["bp_systolic"]
        dbp = clean_v["bp_diastolic"]

        # --- PARADIGM 1: Rules Only ---
        r_tier, _, _, _ = evaluate_hard_escalations(
            spo2, sbp, dbp, pulse, temp, complaint,
            age=age,
            age_months=age_months,
            pregnant=pregnant,
            pregnancy_stage=pregnancy_stage,
            recently_delivered=recently_delivered,
            known_conditions=conds,
            medications=meds
        )
        if r_tier is None:
            r_tier, _ = score_tier_without_hard_escalation(
                spo2, sbp, pulse, temp, complaint, age, age_months
            )
        r_dept = recommend_department(
            complaint, age, gender, r_tier, age_months, pregnant, recently_delivered
        )
        records["rules_only"].append({
            "pred_tier": r_tier,
            "pred_dept": r_dept,
            "abstained": False,
            "gt_tier": gt_tier,
            "gt_emerg": gt_emerg,
            "gt_dept": gt_dept
        })

        # --- PARADIGM 2: Model Only ---
        m_tier, m_conf, _, m_dept = predict_ml_triage(complaint, age, gender, spo2, sbp, dbp, pulse, temp, conds, meds)
        m_tier = m_tier if m_tier is not None else 3
        m_dept = m_dept or "General Medicine"
        records["model_only"].append({
            "pred_tier": m_tier,
            "pred_dept": m_dept,
            "abstained": False,
            "gt_tier": gt_tier,
            "gt_emerg": gt_emerg,
            "gt_dept": gt_dept
        })

        # --- PARADIGM 3: Hybrid (Model + Rules + Abstention) ---
        h_res = run_triage(
            patient_id=enc_id,
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
            medications=meds,
            age_months=age_months,
            pregnant=pregnant,
            pregnancy_stage=pregnancy_stage,
            recently_delivered=recently_delivered
        )
        records["hybrid"].append({
            "pred_tier": h_res["tier"],
            "pred_dept": h_res["department"],
            "abstained": h_res["abstained"],
            "is_escalated": h_res["is_escalated"],
            "gt_tier": gt_tier,
            "gt_emerg": gt_emerg,
            "gt_dept": gt_dept
        })

    # Compute comparative metrics table
    comparison_table = []
    confusion_matrices = {}

    for p in paradigms:
        preds = records[p]
        crit_total = sum(1 for item in preds if item["gt_emerg"] == 1)
        crit_detected = 0
        missed_emergencies = 0
        abstained_count = 0
        total_cost = 0.0
        exact_tier_matches = 0
        dept_matches = 0
        evaluable_tier_count = 0

        cm = {f"Actual L{t}": {f"Pred L{pr}": 0 for pr in range(1, 6)} for t in range(1, 6)}

        for item in preds:
            gt_t = item["gt_tier"]
            pr_t = item["pred_tier"]
            is_crit = item["gt_emerg"] == 1
            is_abstained = item["abstained"]

            if is_abstained:
                abstained_count += 1
                # Abstention maps to nurse review; does NOT penalize under-triage
                total_cost += 0.5  # Mild routing friction cost
                continue

            evaluable_tier_count += 1
            if pr_t == gt_t:
                exact_tier_matches += 1

            if item["pred_dept"] == item["gt_dept"]:
                dept_matches += 1

            # Emergency recall
            if is_crit:
                if pr_t in [1, 2]:
                    crit_detected += 1
                else:
                    missed_emergencies += 1

            # Cost calculation
            total_cost += calculate_asymmetric_cost(gt_t, pr_t)
            if 1 <= gt_t <= 5 and 1 <= pr_t <= 5:
                cm[f"Actual L{gt_t}"][f"Pred L{pr_t}"] += 1

        recall = (crit_detected / max(1, crit_total)) * 100
        abstention_rate = (abstained_count / total_samples) * 100
        coverage = 100.0 - abstention_rate
        avg_cost = total_cost / total_samples
        tier_acc = (exact_tier_matches / max(1, evaluable_tier_count)) * 100
        dept_acc = (dept_matches / max(1, evaluable_tier_count)) * 100

        comparison_table.append({
            "Paradigm": "1. Rules Only" if p == "rules_only" else ("2. Model Only" if p == "model_only" else "3. Hybrid (Model + Rules + Abstention)"),
            "Emergency Recall (Sensitivity)": f"{recall:.1f}%",
            "Missed Emergencies": missed_emergencies,
            "Asymmetric Risk Cost": round(avg_cost, 2),
            "Safety Abstention Rate": f"{abstention_rate:.1f}%",
            "Coverage": f"{coverage:.1f}%",
            "Department Accuracy": f"{dept_acc:.1f}%"
        })

        confusion_matrices[p] = cm

    organizer_output = execute_organizer_evaluate_script()
    org_status = check_organizer_data_status()

    return {
        "is_placeholder": is_placeholder,
        "is_cached": False,
        "organizer_status": org_status,
        "organizer_evaluate_output": organizer_output,
        "total_evaluated": total_samples,
        "comparison_table": pd.DataFrame(comparison_table),
        "hybrid_metrics": {
            "emergency_recall": comparison_table[2]["Emergency Recall (Sensitivity)"],
            "missed_emergencies": comparison_table[2]["Missed Emergencies"],
            "asymmetric_cost_score": comparison_table[2]["Asymmetric Risk Cost"],
            "abstention_rate": comparison_table[2]["Safety Abstention Rate"],
            "coverage": comparison_table[2]["Coverage"],
            "dept_accuracy": comparison_table[2]["Department Accuracy"],
            "confusion_matrix": confusion_matrices["hybrid"]
        }
    }


def simulate_queue_wait_times(num_doctors: int = 2) -> Dict[str, Any]:
    """Replay organizer arrivals and consultation durations under FCFS and tier priority."""
    if num_doctors < 1:
        raise ValueError("num_doctors must be at least 1.")

    arrival_path = ORGANIZER_DATA_DIR / "arrival_log.csv"
    labels_path = ORGANIZER_DATA_DIR / "train_labels.csv"
    if not arrival_path.is_file() or not labels_path.is_file():
        return {
            "available": False,
            "message": "Organizer arrival and label files are required for a measured queue replay.",
        }

    arrivals = pd.read_csv(arrival_path)
    labels = pd.read_csv(labels_path)
    required_arrival_columns = {
        "visit_id", "opd_date", "token_no", "arrival_ts", "consult_minutes",
        "fcfs_consult_start_ts",
    }
    if not required_arrival_columns.issubset(arrivals.columns):
        raise ValueError("Organizer arrival_log.csv is missing required simulation columns.")
    if not {"visit_id", "triage_level", "true_emergency"}.issubset(labels.columns):
        raise ValueError("Organizer train_labels.csv is missing required triage columns.")
    if not arrivals["visit_id"].is_unique or not labels["visit_id"].is_unique:
        raise ValueError("Organizer visit_id values must be unique for queue replay.")

    visits = arrivals.merge(labels, on="visit_id", how="inner", validate="one_to_one")
    if len(visits) != len(labels):
        raise ValueError("Every labeled training visit must have exactly one arrival-log row.")
    visits["arrival_ts"] = pd.to_datetime(visits["arrival_ts"], errors="coerce")
    visits["fcfs_consult_start_ts"] = pd.to_datetime(
        visits["fcfs_consult_start_ts"], errors="coerce"
    )
    visits["consult_minutes"] = pd.to_numeric(visits["consult_minutes"], errors="coerce")
    visits["triage_level"] = pd.to_numeric(visits["triage_level"], errors="coerce")
    if (
        visits[["arrival_ts", "fcfs_consult_start_ts", "consult_minutes", "triage_level"]]
        .isna()
        .any()
        .any()
        or (visits["consult_minutes"] <= 0).any()
        or (~visits["triage_level"].isin([1, 2, 3, 4, 5])).any()
    ):
        raise ValueError("Organizer arrival or triage data contains invalid simulation values.")
    if not (visits["true_emergency"].astype(int) == visits["triage_level"].le(2).astype(int)).all():
        raise ValueError("Organizer emergency labels disagree with triage levels.")

    visits["opd_date"] = pd.to_datetime(visits["opd_date"], errors="coerce").dt.date
    if visits["opd_date"].isna().any():
        raise ValueError("Organizer arrival_log.csv contains invalid OPD dates.")

    fcfs_waits = []
    priority_waits = []
    emergency_fcfs_waits = []
    emergency_priority_waits = []
    replay_error_seconds = []

    for _, day_visits in visits.groupby("opd_date", sort=True):
        day_visits = day_visits.sort_values(
            ["arrival_ts", "token_no"], kind="stable"
        ).reset_index(drop=True)
        arrivals_for_day = [
            (
                row.arrival_ts,
                int(row.triage_level),
                int(row.token_no),
                str(row.visit_id),
                float(row.consult_minutes),
            )
            for row in day_visits.itertuples(index=False)
        ]
        opening = pd.Timestamp(day_visits.iloc[0]["opd_date"]).replace(
            hour=8, minute=0, second=0
        )

        def replay(priority_order: bool) -> Dict[str, float]:
            next_arrival = 0
            now = opening
            waiting = []
            free_doctors = list(range(num_doctors))
            heapq.heapify(free_doctors)
            busy_doctors = []
            waits = {}

            while next_arrival < len(arrivals_for_day) or waiting or busy_doctors:
                while busy_doctors and busy_doctors[0][0] <= now:
                    _, doctor_id = heapq.heappop(busy_doctors)
                    heapq.heappush(free_doctors, doctor_id)
                while (
                    next_arrival < len(arrivals_for_day)
                    and arrivals_for_day[next_arrival][0] <= now
                ):
                    arrival, tier, token, visit_id, service_minutes = arrivals_for_day[next_arrival]
                    order = (tier, arrival, token) if priority_order else (arrival, token)
                    heapq.heappush(waiting, (*order, visit_id, service_minutes))
                    next_arrival += 1

                while free_doctors and waiting:
                    doctor_id = heapq.heappop(free_doctors)
                    item = heapq.heappop(waiting)
                    arrival = item[1] if priority_order else item[0]
                    visit_id = item[-2]
                    service_minutes = item[-1]
                    waits[visit_id] = (now - arrival).total_seconds() / 60.0
                    finish = now + pd.Timedelta(minutes=service_minutes)
                    heapq.heappush(busy_doctors, (finish, doctor_id))

                if next_arrival < len(arrivals_for_day) or busy_doctors:
                    upcoming = []
                    if next_arrival < len(arrivals_for_day):
                        upcoming.append(arrivals_for_day[next_arrival][0])
                    if busy_doctors:
                        upcoming.append(busy_doctors[0][0])
                    now = min(upcoming)
            return waits

        fcfs_starts = replay(priority_order=False)
        priority_starts = replay(priority_order=True)
        for row in day_visits.itertuples(index=False):
            visit_id = str(row.visit_id)
            observed_wait = (
                row.fcfs_consult_start_ts - row.arrival_ts
            ).total_seconds() / 60.0
            fcfs_wait = fcfs_starts[visit_id]
            priority_wait = priority_starts[visit_id]
            if observed_wait < 0 or fcfs_wait < 0 or priority_wait < 0:
                raise ValueError("Queue replay produced a consultation start before arrival.")
            replay_error_seconds.append(
                abs((row.fcfs_consult_start_ts - (
                    row.arrival_ts + pd.Timedelta(minutes=fcfs_wait)
                )).total_seconds())
            )
            fcfs_waits.append(fcfs_wait)
            priority_waits.append(priority_wait)
            if int(row.triage_level) <= 2:
                emergency_fcfs_waits.append(fcfs_wait)
                emergency_priority_waits.append(priority_wait)

    mean_fcfs_emergency = float(np.mean(emergency_fcfs_waits))
    mean_priority_emergency = float(np.mean(emergency_priority_waits))
    reduction = (
        (mean_fcfs_emergency - mean_priority_emergency) / mean_fcfs_emergency * 100.0
        if mean_fcfs_emergency > 0
        else 0.0
    )
    return {
        "available": True,
        "num_doctors": num_doctors,
        "patients_simulated": int(len(visits)),
        "emergency_patients": len(emergency_fcfs_waits),
        "fcfs_emergency_wait_mins": round(mean_fcfs_emergency, 1),
        "cpg_emergency_wait_mins": round(mean_priority_emergency, 1),
        "emergency_wait_reduction_pct": round(reduction, 1),
        "fcfs_overall_wait_mins": round(float(np.mean(fcfs_waits)), 1),
        "cpg_overall_wait_mins": round(float(np.mean(priority_waits)), 1),
        "official_fcfs_replay_mean_error_seconds": round(float(np.mean(replay_error_seconds)), 2),
        "data_source": "Organizer arrival_log.csv and train_labels.csv",
        "interpretation": (
            "Counterfactual priority replay uses organizer ground-truth tiers, not predictions; "
            "it is not a measured estimate of this model's wait-time impact."
        ),
        "assumptions": [
            f"Replays {len(visits):,} labeled synthetic visits across their recorded OPD dates.",
            f"Uses {num_doctors} non-preemptive doctors, measured consultation minutes, and recorded arrivals.",
            "FCFS baseline is replayed from arrival and service data; organizer FCFS starts are cross-checked.",
            "Priority order is ground-truth tier 1 to 5, then arrival time; no nurse-review cases are present.",
        ],
    }
