"""CarePriority Gov - Clinical Shift Handover Summary Module
=========================================================
Generates rule-based clinical SBAR (Situation, Background, Assessment, Recommendation)
shift handover reports and matching tabular encounter CSV exports.

Ensures strict parity:
- The total patient count and tier counts in the .txt clinical handover report
  identically match the record counts in the exported .csv dataset.
"""

import io
import csv
from datetime import datetime
from typing import List, Dict, Any, Tuple
import db


def compute_shift_metrics(encounters: List[Dict[str, Any]], num_doctors: int = 2) -> Dict[str, Any]:
    """Computes comprehensive shift statistics and clinical safety indicators."""
    total = len(encounters)
    now = datetime.now()

    tier_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, "abstained": 0}
    status_counts = {"Waiting": 0, "Awaiting Nurse Review": 0, "Completed": 0, "In Progress": 0, "Other": 0}
    escalation_count = 0
    override_count = 0
    retriage_due_count = 0
    total_wait_minutes = 0
    waiting_count = 0
    high_risk_waiting = []
    department_counts = {}

    for enc in encounters:
        # Tier counts
        abstained = enc.get("abstained", False)
        t = enc.get("tier")
        if abstained or t is None:
            tier_counts["abstained"] += 1
        elif t in tier_counts:
            tier_counts[t] += 1

        # Department
        dept = enc.get("department", "General Medicine")
        department_counts[dept] = department_counts.get(dept, 0) + 1

        # Escalations
        if enc.get("is_escalated"):
            escalation_count += 1

        # Nurse overrides in audit log
        audit = enc.get("audit", [])
        for entry in audit:
            if "Override" in entry.get("action", "") or "Nurse Override" in entry.get("action", ""):
                override_count += 1

        # Status & Waiting times
        st_val = enc.get("status", "Waiting")
        if st_val in status_counts:
            status_counts[st_val] += 1
        else:
            status_counts["Other"] += 1

        created_at_str = enc.get("created_at", "")
        wait_mins = 0
        if created_at_str:
            try:
                dt = datetime.fromisoformat(created_at_str)
                wait_mins = max(0, int((now - dt).total_seconds() / 60))
            except Exception:
                wait_mins = 0

        is_waiting = st_val in ["Waiting", "Awaiting Nurse Review"]
        if is_waiting:
            waiting_count += 1
            total_wait_minutes += wait_mins

            # Check if retriage is due
            is_due, _, _ = db.check_retriage_due(enc, now=now)
            if is_due:
                retriage_due_count += 1

            # High risk cases (L1, L2, or Abstained/Needs Review)
            if abstained or t in [1, 2]:
                high_risk_waiting.append({
                    "encounter_id": enc.get("encounter_id", "N/A"),
                    "tier": t,
                    "tier_name": "Needs Review" if abstained or t is None else enc.get("tier_name", f"Level {t}"),
                    "department": dept,
                    "wait_minutes": wait_mins,
                    "retriage_due": is_due,
                    "reasons": enc.get("reasons", [])[:2],
                    "red_flags": enc.get("red_flags", [])[:3]
                })

    avg_wait = round(total_wait_minutes / max(1, waiting_count), 1) if waiting_count > 0 else 0.0

    return {
        "total_encounters": total,
        "waiting_count": waiting_count,
        "avg_wait_minutes": avg_wait,
        "tier_counts": tier_counts,
        "status_counts": status_counts,
        "department_counts": department_counts,
        "escalation_count": escalation_count,
        "override_count": override_count,
        "retriage_due_count": retriage_due_count,
        "high_risk_waiting": high_risk_waiting,
        "num_doctors": num_doctors,
        "generated_at": now.strftime("%Y-%m-%d %H:%M:%S")
    }


def generate_shift_handover_txt(encounters: List[Dict[str, Any]], num_doctors: int = 2) -> str:
    """Generates a structured human-readable SBAR shift report string."""
    m = compute_shift_metrics(encounters, num_doctors=num_doctors)
    tot = max(1, m["total_encounters"])
    tc = m["tier_counts"]

    lines = []
    lines.append("=" * 80)
    lines.append("CAREPRIORITY GOV - CLINICAL SHIFT HANDOVER SUMMARY (SBAR)")
    lines.append("Government District Hospital OPD & Emergency Command Center")
    lines.append(f"Shift Report Generated: {m['generated_at']}")
    lines.append(f"Active Duty Physicians: {num_doctors} | Triage Engine: Rule-Guardrailed AI (5 Tiers)")
    lines.append("=" * 80)
    lines.append("")

    # S: Situation
    lines.append("[S] SITUATION (OPD Capacity & Active Workload)")
    lines.append("-" * 80)
    lines.append(f"Total Patient Encounters Processed in Shift : {m['total_encounters']}")
    lines.append(f"Active Patients Currently in Queue          : {m['waiting_count']}")
    lines.append(f"Average Patient Wait Time                   : {m['avg_wait_minutes']} minutes")
    lines.append(f"Patients Awaiting Nurse Review / Incomplete : {m['status_counts'].get('Awaiting Nurse Review', 0)}")
    lines.append(f"Physician Workload Ratio                    : {round(m['waiting_count'] / max(1, num_doctors), 1)} waiting patients per on-duty doctor")
    lines.append("")

    # B: Background
    lines.append("[B] BACKGROUND (Clinical Severity Breakdown)")
    lines.append("-" * 80)
    lines.append(f"  Level 1 (Resuscitation - 0 min)   : {tc[1]:3d} ({round(tc[1]/tot*100, 1)}%)")
    lines.append(f"  Level 2 (Emergent - <15 min)       : {tc[2]:3d} ({round(tc[2]/tot*100, 1)}%)")
    lines.append(f"  Level 3 (Urgent - <45 min)         : {tc[3]:3d} ({round(tc[3]/tot*100, 1)}%)")
    lines.append(f"  Level 4 (Less Urgent - <90 min)    : {tc[4]:3d} ({round(tc[4]/tot*100, 1)}%)")
    lines.append(f"  Level 5 (Non-Urgent - <180 min)    : {tc[5]:3d} ({round(tc[5]/tot*100, 1)}%)")
    lines.append(f"  Safety Abstentions (Nurse Review)  : {tc['abstained']:3d} ({round(tc['abstained']/tot*100, 1)}%)")
    lines.append("")
    lines.append(f"Hard-Coded Physiological Escalations Triggered : {m['escalation_count']}")
    lines.append(f"Patients Exceeding Tier Wait Threshold (Due)   : {m['retriage_due_count']}")
    lines.append(f"Clinical Nurse Overrides Logged with Rationale : {m['override_count']}")
    lines.append("")

    # A: Assessment
    lines.append("[A] ASSESSMENT (Active High-Risk Queue & Critical Alerts)")
    lines.append("-" * 80)
    if m["high_risk_waiting"]:
        lines.append(f"WARNING: {len(m['high_risk_waiting'])} critical/emergent patients are currently waiting:")
        for idx, hr in enumerate(m["high_risk_waiting"][:10], 1):
            rf_str = ", ".join(hr["red_flags"]) if hr["red_flags"] else "None flagged"
            due_str = " [RETRIAGE OVERDUE]" if hr["retriage_due"] else ""
            lines.append(f"  {idx}. [{hr['encounter_id']}] {hr['tier_name']} | Dept: {hr['department']} | Wait: {hr['wait_minutes']}m{due_str}")
            lines.append(f"     Red flags / Triggers: {rf_str}")
    else:
        lines.append("No Level 1 or Level 2 patients currently waiting. Queue is clinically stable.")
    lines.append("")

    # R: Recommendation
    lines.append("[R] RECOMMENDATION (Operational Handover Directives)")
    lines.append("-" * 80)
    lines.append("1. Incoming charge nurse must immediately verify and assign the active Level 1/2 cases.")
    lines.append("2. Perform repeat vital sign assessments on any patient waiting past their tier threshold.")
    lines.append("3. Review any unreviewed Safety Abstentions at the intake desk.")
    lines.append("4. Verify crash cart readiness and medical oxygen cylinder pressure.")
    lines.append("=" * 80)
    lines.append("DISCLAIMER: Independent demo software. Not validated for official clinical care.")
    lines.append("=" * 80)

    return "\n".join(lines)


def generate_shift_handover_csv(encounters: List[Dict[str, Any]]) -> str:
    """Generates an encounter-level CSV string matching the shift metrics."""
    output = io.StringIO()
    fieldnames = [
        "encounter_id",
        "created_at",
        "patient_name",
        "age",
        "gender",
        "tier",
        "tier_name",
        "department",
        "status",
        "abstained",
        "abstention_reason",
        "spo2",
        "bp_systolic",
        "bp_diastolic",
        "pulse",
        "temperature",
        "is_escalated",
        "escalation_rule",
        "audit_override_count"
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()

    for enc in encounters:
        v = enc.get("vitals", {}) or {}
        audit = enc.get("audit", []) or []
        overrides = sum(1 for a in audit if "Override" in a.get("action", ""))

        row = {
            "encounter_id": enc.get("encounter_id", ""),
            "created_at": enc.get("created_at", ""),
            "patient_name": enc.get("patient_name", ""),
            "age": enc.get("age", ""),
            "gender": enc.get("gender", "Unknown"),
            "tier": enc.get("tier", ""),
            "tier_name": enc.get("tier_name", "Needs Review" if enc.get("abstained") else ""),
            "department": enc.get("department", "General Medicine"),
            "status": enc.get("status", "Waiting"),
            "abstained": enc.get("abstained", False),
            "abstention_reason": enc.get("abstention_reason", ""),
            "spo2": v.get("spo2", ""),
            "bp_systolic": v.get("bp_systolic", ""),
            "bp_diastolic": v.get("bp_diastolic", ""),
            "pulse": v.get("pulse", ""),
            "temperature": v.get("temperature", ""),
            "is_escalated": enc.get("is_escalated", False),
            "escalation_rule": enc.get("rule_triggered") or enc.get("escalation_rule", ""),
            "audit_override_count": overrides
        }
        writer.writerow(row)

    return output.getvalue()
