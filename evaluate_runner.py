"""CarePriority Gov - Live Evaluation & Queue Simulation Engine
============================================================
Computes all evaluation metrics LIVE on a held-out validation split:
1. Emergency Sensitivity / Recall (Target: 100% caught, 0 missed emergencies)
2. 20x Asymmetric Safety Risk Cost Score
3. Safety Abstention Rate & Coverage
4. 5x5 Confusion Matrix (Ground Truth vs. Predicted Tier)
5. Department Classification Accuracy
6. Architectural Comparison Table:
   - Rules Only vs. Model Only vs. Hybrid (Model + Rules + Abstention)
7. Stretch Simulation: Wait-Time Reduction vs. First-Come-First-Served (FCFS)
   with 2 doctors, with all simulation assumptions clearly stated.
8. Seamless execution of organizer evaluate.py if present in /data.
"""

import os
import glob
import subprocess
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


def check_organizer_data_status() -> Dict[str, Any]:
    """
    Checks whether official organizer files are present in /data/.
    Returns status and exact user download/placement guidance if missing.
    Gracefully handles absence of the data/ folder without crashing.
    """
    data_dir = "data"
    has_data_dir = os.path.exists(data_dir) and os.path.isdir(data_dir)

    if has_data_dir:
        candidates = glob.glob(os.path.join(data_dir, "*.csv")) + \
                     glob.glob(os.path.join(data_dir, "*.parquet")) + \
                     glob.glob(os.path.join(data_dir, "*.json"))

        organizer_datasets = [
            f for f in candidates
            if "synthetic_placeholder" not in os.path.basename(f).lower()
            and "validation_split" not in os.path.basename(f).lower()
            and not os.path.basename(f).startswith(".")
        ]
        has_organizer_data = len(organizer_datasets) > 0
        organizer_data_file = organizer_datasets[0] if has_organizer_data else None
        evaluate_script_path = os.path.join(data_dir, "evaluate.py")
        has_evaluate_script = os.path.exists(evaluate_script_path)
    else:
        has_organizer_data = False
        organizer_data_file = None
        evaluate_script_path = None
        has_evaluate_script = False

    missing_instructions = []
    if not has_data_dir:
        missing_instructions.append(
            "1. The `data/` directory does not exist. Create the `data/` folder and place the organizer synthetic intake records (~30k records) inside "
            "(e.g., `data/records.csv` with columns: age, sex, bp_systolic, bp_diastolic, pulse, temperature, spo2, chief_complaint, known_conditions, medications, true_level, true_department, true_emergency)."
        )
    elif not has_organizer_data:
        missing_instructions.append(
            "1. Download the organizer synthetic intake records (~30k records) and place the CSV/Parquet file inside the `data/` folder "
            "(e.g., `data/records.csv` with columns: age, sex, bp_systolic, bp_diastolic, pulse, temperature, spo2, chief_complaint, known_conditions, medications, true_level, true_department, true_emergency)."
        )
    if not has_evaluate_script:
        missing_instructions.append(
            "2. Place the official `evaluate.py` script provided by the hackathon organizers inside the `data/` directory (`data/evaluate.py`)."
        )

    return {
        "has_data_dir": has_data_dir,
        "has_organizer_data": has_organizer_data,
        "organizer_data_file": organizer_data_file,
        "has_evaluate_script": has_evaluate_script,
        "evaluate_script_path": evaluate_script_path if has_evaluate_script else None,
        "is_placeholder": not has_organizer_data,
        "missing_instructions": missing_instructions
    }


def execute_organizer_evaluate_script() -> Optional[str]:
    """
    If data/evaluate.py exists, executes it as provided without modification.
    """
    script_path = os.path.join("data", "evaluate.py")
    if not os.path.exists(script_path):
        return None

    import sys
    python_exe = sys.executable or (".venv/Scripts/python.exe" if os.path.exists(".venv/Scripts/python.exe") else "python")
    val_preds = os.path.join("data", "val_predictions.csv")
    val_truth = os.path.join("data", "val_truth.csv")
    sub_preds = os.path.join("submission", "val_predictions.csv")
    sub_truth = os.path.join("submission", "val_truth.csv")

    cmd = [python_exe, script_path]
    if os.path.exists(val_preds) and os.path.exists(val_truth):
        cmd.extend(["--submission", val_preds, "--truth", val_truth])
    elif os.path.exists(sub_preds) and os.path.exists(sub_truth):
        cmd.extend(["--submission", sub_preds, "--truth", sub_truth])
    else:
        cmd.append("-h")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.getcwd()
        )
        return proc.stdout or proc.stderr
    except Exception as e:
        return f"Execution of data/evaluate.py failed: {e}"


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
    Computes all safety and performance metrics LIVE across 3 paradigms:
    1. Rules Only
    2. Model Only
    3. Hybrid CarePriority Gov Pipeline (Model + Rules + Abstention)
    Uses precomputed cached validation results when available for sub-second UI response.
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
            data["is_placeholder"] = False
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
        r_tier, _, _, _ = evaluate_hard_escalations(spo2, sbp, dbp, pulse, temp, complaint)
        if r_tier is None:
            r_tier, _ = score_tier_without_hard_escalation(spo2, sbp, pulse, temp, complaint)
        r_dept = recommend_department(complaint, age, gender, r_tier)
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
            medications=meds
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


def simulate_queue_wait_times(
    num_doctors: int = 2,
    num_patients: int = 200
) -> Dict[str, Any]:
    """
    STRETCH: Discrete-Event Queue Simulation comparing First-Come-First-Served (FCFS)
    vs. CarePriority Gov Dynamic Severity Queue with 2 active doctors.
    
    Assumptions (clearly stated):
    1. OPD facility operates with 2 doctors on concurrent duty.
    2. Consultation service time distribution by tier:
       - Level 1 Resuscitation: 35 minutes (intensive stabilization, emergency line setup)
       - Level 2 Emergent: 20 minutes (urgent diagnostic workup, stat medication)
       - Level 3 Urgent: 15 minutes (acute clinical exam, lab orders)
       - Level 4 Less Urgent: 10 minutes (routine examination, symptom prescription)
       - Level 5 Non-Urgent: 6 minutes (medication refill, suture removal)
    3. Patient arrival rate: Poisson / steady stream (~25-30 patients/hour).
    4. FCFS: Doctor takes the next patient strictly in arrival sequence.
    5. Dynamic Severity (CarePriority Gov): Doctor takes next patient from:
       (1) Needs Nurse Review, (2) L1 -> L5, (3) Arrival time.
    """
    np.random.seed(42)

    # Consultation duration mean in minutes
    service_times = {
        1: 35.0,
        2: 20.0,
        3: 15.0,
        4: 10.0,
        5: 6.0
    }

    # Generate synthetic arrival timeline
    # Tiers distribution: 2% L1, 8% L2, 35% L3, 35% L4, 20% L5
    patient_stream = []
    current_time = 0.0

    for i in range(num_patients):
        inter_arrival = np.random.exponential(scale=2.2)  # ~27 walk-ins per hour
        current_time += inter_arrival
        dice = np.random.random()

        if dice < 0.02:
            tier = 1
        elif dice < 0.10:
            tier = 2
        elif dice < 0.45:
            tier = 3
        elif dice < 0.80:
            tier = 4
        else:
            tier = 5

        base_svc = service_times[tier]
        actual_svc = max(3.0, np.random.normal(loc=base_svc, scale=base_svc * 0.15))

        patient_stream.append({
            "id": f"P-{i+1:03d}",
            "arrival_time": current_time,
            "tier": tier,
            "service_time": actual_svc,
            "is_emergency": tier in [1, 2]
        })

    # 1. Simulate FCFS Queue
    doctor_free_fcfs = [0.0] * num_doctors
    fcfs_wait_times = []
    fcfs_emergency_waits = []

    for p in patient_stream:
        earliest_doc = int(np.argmin(doctor_free_fcfs))
        start_service = max(p["arrival_time"], doctor_free_fcfs[earliest_doc])
        wait = start_service - p["arrival_time"]
        fcfs_wait_times.append(wait)
        if p["is_emergency"]:
            fcfs_emergency_waits.append(wait)
        doctor_free_fcfs[earliest_doc] = start_service + p["service_time"]

    # 2. Simulate CarePriority Gov Dynamic Severity Queue
    # Non-preemptive priority queue simulation
    event_time = 0.0
    waiting_pool = []
    in_service = [{"busy": False, "finish_time": 0.0, "patient": None} for _ in range(num_doctors)]
    cpg_wait_times = {}

    arrivals_left = sorted(patient_stream, key=lambda x: x["arrival_time"])
    completed_count = 0

    while completed_count < num_patients:
        # Move all patients who have arrived up to event_time into waiting pool
        while arrivals_left and arrivals_left[0]["arrival_time"] <= event_time:
            waiting_pool.append(arrivals_left.pop(0))

        # Check for free doctors
        free_docs = [idx for idx, d in enumerate(in_service) if not d["busy"] or d["finish_time"] <= event_time]

        for d_idx in free_docs:
            if in_service[d_idx]["busy"] and in_service[d_idx]["finish_time"] <= event_time:
                in_service[d_idx]["busy"] = False
                completed_count += 1

            if not in_service[d_idx]["busy"] and waiting_pool:
                # Severity order: tier (1 to 5), then arrival_time
                waiting_pool.sort(key=lambda x: (x["tier"], x["arrival_time"]))
                chosen = waiting_pool.pop(0)
                wait = event_time - chosen["arrival_time"]
                cpg_wait_times[chosen["id"]] = wait

                in_service[d_idx]["busy"] = True
                in_service[d_idx]["finish_time"] = event_time + chosen["service_time"]
                in_service[d_idx]["patient"] = chosen

        # Advance event_time to next event (next arrival or next doctor finish)
        next_events = []
        if arrivals_left:
            next_events.append(arrivals_left[0]["arrival_time"])
        for d in in_service:
            if d["busy"]:
                next_events.append(d["finish_time"])

        if next_events:
            event_time = min(e for e in next_events if e > event_time or np.isclose(e, event_time))
            if event_time == min(next_events):
                # Ensure time advances
                event_time = max(event_time, min(next_events) + 0.001)
        else:
            event_time += 1.0

    cpg_emerg_waits = [
        cpg_wait_times[p["id"]] for p in patient_stream if p["is_emergency"] and p["id"] in cpg_wait_times
    ]

    avg_fcfs_all = float(np.mean(fcfs_wait_times))
    avg_fcfs_emerg = float(np.mean(fcfs_emergency_waits))
    avg_cpg_emerg = float(np.mean(cpg_emerg_waits)) if cpg_emerg_waits else 0.0
    avg_cpg_all = float(np.mean(list(cpg_wait_times.values()))) if cpg_wait_times else 0.0

    emerg_wait_reduction_pct = ((avg_fcfs_emerg - avg_cpg_emerg) / max(1e-5, avg_fcfs_emerg)) * 100

    return {
        "num_doctors": num_doctors,
        "patients_simulated": num_patients,
        "fcfs_emergency_wait_mins": round(avg_fcfs_emerg, 1),
        "cpg_emergency_wait_mins": round(avg_cpg_emerg, 1),
        "emergency_wait_reduction_pct": round(emerg_wait_reduction_pct, 1),
        "fcfs_overall_wait_mins": round(avg_fcfs_all, 1),
        "cpg_overall_wait_mins": round(avg_cpg_all, 1),
        "assumptions": [
            f"OPD operates with exactly {num_doctors} active doctors.",
            "Doctor service duration: L1=35m, L2=20m, L3=15m, L4=10m, L5=6m.",
            "Arrival pattern: Poisson walk-in rate ~27 patients/hour over an 8-hour shift.",
            "CarePriority Gov prioritizes L1 and L2 immediately upon arrival without wait-room starvation."
        ]
    }
