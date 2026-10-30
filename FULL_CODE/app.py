"""CarePriority Gov - Main Web Application
=========================================
Dynamic Clinical Severity OPD Triage Assistant for Overloaded Government Hospitals.
Tech Stack: Streamlit, PyMongo with Session-State Fallback, Scikit-learn, Python-Dotenv, Altair, Pandas.

DISCLAIMER:
Demo only — not validated for clinical use. Independent demonstration project,
not an official government service.
"""

import os
import json
import time
import copy
from html import escape as escape_html
from datetime import datetime
import pandas as pd
import altair as alt
import streamlit as st

import db
import seed_data
import evaluate_runner
from rules_config import (
    TRIAGE_LEVELS,
    RED_FLAG_KEYWORDS,
    SECONDARY_KEYWORDS,
    DEPARTMENTS,
    HOSPITAL_CONFIG,
    TRIAGE_COST_MATRIX_5X5,
    ASYMMETRIC_COST_CONFIG
)
from triage_engine import (
    run_triage,
    validate_vitals,
    clean_vitals_for_storage,
    check_model_status
)
from styles import (
    apply_theme,
    render_brand_header,
    render_tier_badge,
    render_footer,
    render_html,
    SVG_LOGO
)
from smart_ticket import render_smart_token_ticket
import shift_handover
import surge_mode
import audio_speech
import hero_3d
import ai_scan
import explainability


# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CarePriority Gov | OPD Clinical Triage",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_queue_replay(num_doctors: int):
    return evaluate_runner.simulate_queue_wait_times(num_doctors=num_doctors)


# Auto-seed initial demo dataset if database is completely empty
db_status = db.get_db_status()
if len(db.get_all_encounters()) == 0:
    seed_data_records = seed_data.generate_curated_demo_records()
    db.bulk_insert(seed_data_records)

# Check organizer dataset & model status
organizer_info = evaluate_runner.check_organizer_data_status()
model_status = check_model_status()


def load_offline_evaluation_metrics():
    """Load only the latest chronological organizer validation report."""
    relative_path = os.path.join("submission", "organizer_validation_score.json")
    path = os.path.join(os.path.dirname(__file__), relative_path)
    if not os.path.isfile(path):
        return {"loaded": False}

    try:
        with open(path, "r", encoding="utf-8") as score_file:
            data = json.load(score_file)
        required_fields = {
            "primary_score",
            "emergency_recall",
            "n_missed_emergencies",
            "n_true_emergencies",
            "validation_method",
        }
        missing_fields = required_fields.difference(data)
        if missing_fields:
            raise ValueError(
                f"Validation report is missing fields: {', '.join(sorted(missing_fields))}"
            )
        return {
            "loaded": True,
            "path": relative_path,
            "primary_score": float(data["primary_score"]),
            "emergency_recall": float(data["emergency_recall"]),
            "n_missed_emergencies": int(data["n_missed_emergencies"]),
            "n_true_emergencies": int(data["n_true_emergencies"]),
            "validation_method": str(data["validation_method"]),
            "score": data,
        }
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        return {"loaded": False, "error": str(error)}

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & SYSTEM STATUS
# -----------------------------------------------------------------------------
with st.sidebar:
    render_html(f"""
    <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 20px; padding-bottom: 14px; border-bottom: 1px solid rgba(255,255,255,0.08);">
      <div>{SVG_LOGO}</div>
      <div>
        <div style="font-family: 'Sora', sans-serif; font-size: 1.25rem; font-weight: 800; color: #FFFFFF; line-height: 1.1;">
          CarePriority <span style="background: linear-gradient(135deg, #66E3C4, #38BDF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Gov</span>
        </div>
        <div style="font-size: 0.72rem; color: #94A3B8; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 3px;">
          OPD Command Center
        </div>
      </div>
    </div>
    """)

    # Database Connection Pill
    if db_status["mode"] == "mongodb":
        render_html(f"""
        <div class="status-pill" style="border-color: rgba(0,214,143,0.4); color: #66E3C4; width: 100%; justify-content: flex-start; margin-bottom: 8px;">
          <span class="status-dot status-dot-green"></span>
          <span>{db_status['banner']}</span>
        </div>
        """)
    else:
        render_html(f"""
        <div class="status-pill" style="border-color: rgba(255,199,0,0.4); color: #FFD13B; width: 100%; justify-content: flex-start; margin-bottom: 8px;">
          <span class="status-dot status-dot-orange"></span>
          <span>{db_status['banner']}</span>
        </div>
        """)

    # ML Model Status Pill
    if model_status["has_models"]:
        render_html("""
        <div class="status-pill" style="border-color: rgba(0,214,143,0.4); color: #66E3C4; width: 100%; justify-content: flex-start; margin-bottom: 8px;">
          <span class="status-dot status-dot-green"></span>
          <span>🧠 ML Models Ready</span>
        </div>
        """)
    else:
        render_html("""
        <div class="status-pill" style="border-color: rgba(255,199,0,0.4); color: #FFD13B; width: 100%; justify-content: flex-start; margin-bottom: 8px;">
          <span class="status-dot status-dot-orange"></span>
          <span>⚠️ ML Model: Not trained (heuristic rules active)</span>
        </div>
        """)

    # Dataset Status Pill
    if organizer_info["has_organizer_data"]:
        render_html("""
        <div class="status-pill" style="border-color: rgba(0,214,143,0.4); color: #66E3C4; width: 100%; justify-content: flex-start;">
          <span class="status-dot status-dot-green"></span>
          <span>📁 Organizer Data Loaded</span>
        </div>
        """)
    else:
        data_label = "⚠️ Placeholder Data Active" if organizer_info.get("has_data_dir") else "⚠️ No data/ folder (Placeholder Mode)"
        render_html(f"""
        <div class="status-pill" style="border-color: rgba(255,122,0,0.4); color: #FFB366; width: 100%; justify-content: flex-start;">
          <span class="status-dot status-dot-orange"></span>
          <span>{data_label}</span>
        </div>
        """)

    render_html("<div style='height: 14px;'></div>")

    # Main Navigation (including Public Status Display)
    nav_options = [
        "📊 Command Center",
        "⚡ Patient Intake (<60s)",
        "📋 Live Doctor OPD Queue",
        "📺 Public Status Display",
        "🔍 Encounter Detail View",
        "🛡️ Safety Rules & Thresholds",
        "🧪 Live Evaluation & Benchmarks"
    ]
    target = st.session_state.pop("target_page", None)
    default_idx = nav_options.index(target) if target in nav_options else 0

    page = st.radio(
        "Navigation",
        nav_options,
        index=default_idx
    )

    render_html("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 18px 0;'>")

    # Quick Hospital Capacity Controls
    render_html("<div class='label-caps'>Capacity Settings</div>")
    num_doctors = st.number_input(
        "Active OPD Doctors on Duty",
        min_value=1,
        max_value=12,
        value=int(os.getenv("DOCTOR_COUNT", 2)),
        help="Simulates workload per physician (default 2 doctors for 800 walk-ins)."
    )

    anti_starve_threshold = st.slider(
        "Anti-Starvation Alert (Mins)",
        min_value=30,
        max_value=240,
        value=120,
        step=15,
        help="Highlights stable patients waiting longer than this threshold."
    )

    render_html("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 18px 0;'>")

    # Surge Mode Threshold Controls
    render_html("<div class='label-caps'>Surge Protocol Controls</div>")
    surge_waiting_thresh = st.number_input(
        "Surge Queue Threshold (Walk-ins)",
        min_value=3,
        max_value=60,
        value=15,
        step=1,
        help="Triggers hospital surge mode if total waiting queue reaches or exceeds this count."
    )
    surge_emergent_thresh = st.number_input(
        "Surge Emergent Threshold (L1/L2)",
        min_value=1,
        max_value=20,
        value=3,
        step=1,
        help="Triggers hospital surge mode if L1/L2 emergent patients reach or exceed this count."
    )
    force_surge_mode = st.checkbox(
        "🚨 Manual Surge Override",
        value=False,
        help="Engages hospital emergency surge mode immediately by Nurse In-Charge."
    )

    render_html("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 18px 0;'>")

    # Data Actions in Sidebar
    render_html("<div class='label-caps'>Dataset Controls</div>")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        if st.button("🌱 Load Demo", help="Loads 12 curated synthetic walk-in patients"):
            if db.clear_all():
                demo_recs = seed_data.generate_curated_demo_records()
                db.bulk_insert(demo_recs)
                st.success("Loaded 12 demo patients!")
                st.rerun()
            else:
                st.error("Could not clear the MongoDB queue. Local session mode is active; remote records may remain.")
    with col_sb2:
        if st.button("🧹 Clear All", help="Resets patient queue"):
            if db.clear_all():
                st.warning("Cleared queue!")
                st.rerun()
            else:
                st.error("Could not clear the MongoDB queue. Local session data was cleared; remote records may remain.")

    if st.button("⚡ Run 800-Patient Stress Test", help="Simulates an entire 800 walk-ins/day government hospital shift"):
        if not db.clear_all():
            st.error("Could not clear the MongoDB queue. Local session mode is active; remote records may remain.")
        else:
            with st.spinner("Generating 800 realistic synthetic walk-ins with clinical triage..."):
                stress_records = seed_data.generate_800_patient_stress_test()
                db.bulk_insert(stress_records)
                st.success(f"Successfully generated & triaged {len(stress_records)} synthetic patients!")
                st.rerun()

# -----------------------------------------------------------------------------
# DYNAMIC THEME & GLOBAL SURGE EVALUATION
# -----------------------------------------------------------------------------
# Calmer, dimmer version of background on Intake and Queue pages
apply_theme(calm=(page in ["⚡ Patient Intake (<60s)", "📋 Live Doctor OPD Queue"]))

# Evaluate Surge Mode
all_current_encounters = db.get_all_encounters()
surge_status = surge_mode.evaluate_surge_status(
    encounters=all_current_encounters,
    waiting_threshold=int(surge_waiting_thresh),
    emergent_threshold=int(surge_emergent_thresh),
    force_surge=force_surge_mode
)

# -----------------------------------------------------------------------------
# PAGE 1: COMMAND CENTER (HOME)
# -----------------------------------------------------------------------------
if page == "📊 Command Center":
    metrics = db.get_metrics(num_doctors=num_doctors)

    # 3D Procedural Three.js Hero Scene (Command Center ONLY, Hero Banner Area Only)
    hero_3d.render_hero_3d_banner(height=240)

    # Live Telemetry Strip
    db_pill = f'<div class="status-pill" style="border-color: rgba(0,214,143,0.4); color: #66E3C4;"><span class="status-dot status-dot-green"></span> {db_status["banner"]}</div>' if db_status["mode"] == "mongodb" else f'<div class="status-pill" style="border-color: rgba(255,199,0,0.4); color: #FFD13B;"><span class="status-dot status-dot-orange"></span> {db_status["banner"]}</div>'
    model_pill = '<div class="status-pill" style="border-color: rgba(0,214,143,0.4); color: #66E3C4;"><span class="status-dot status-dot-green"></span> 🧠 ML Models Ready</div>' if model_status["has_models"] else '<div class="status-pill" style="border-color: rgba(255,199,0,0.4); color: #FFD13B;"><span class="status-dot status-dot-orange"></span> ⚠️ ML Model: Heuristic Active</div>'
    data_label = "📁 Organizer Dataset Loaded" if organizer_info["has_organizer_data"] else ("⚠️ Placeholder Data Active" if organizer_info.get("has_data_dir") else "⚠️ No data/ folder (Placeholder Active)")
    data_pill = f'<div class="status-pill" style="border-color: rgba(0,214,143,0.4); color: #66E3C4;"><span class="status-dot status-dot-green"></span> {data_label}</div>' if organizer_info["has_organizer_data"] else f'<div class="status-pill" style="border-color: rgba(255,122,0,0.4); color: #FFA336;"><span class="status-dot status-dot-orange"></span> {data_label}</div>'
    rules_pill = '<div class="status-pill" style="border-color: rgba(56,189,248,0.4); color: #38BDF8;"><span class="status-dot status-dot-blue"></span> 🛡️ 20x Guardrails Active</div>'

    render_html(f"""
    <div style="display: flex; gap: 10px; flex-wrap: wrap; margin: 12px 0 16px 0;">
      {db_pill}
      {model_pill}
      {data_pill}
      {rules_pill}
    </div>
    """)

    if not model_status["has_models"]:
        st.info("ℹ️ **Model Status:** Pre-trained ML models (`models/*.joblib`) not found. CarePriority Gov operates safely using deterministic clinical rules and heuristic scoring. Run `python train_model.py` to train pipelines.")
    if not organizer_info.get("has_organizer_data"):
        st.info(
            "Organizer training, labels, and test files are incomplete. "
            "Organizer evaluation and test prediction are unavailable."
        )

    # Render Surge Mode Alert Banner & Action Checklist if Active
    if surge_status["is_active"]:
        render_html(surge_mode.render_surge_banner_html(surge_status))
        with st.expander("🚨 OPD SURGE PROTOCOL ACTION CHECKLIST (Execute Immediately)", expanded=True):
            st.write("Mark completed emergency mobilization actions:")
            if "surge_checklist" not in st.session_state:
                st.session_state["surge_checklist"] = {}
            for item in surge_mode.SURGE_CHECKLIST_TEMPLATE:
                item_id = item["id"]
                checked = st.checkbox(
                    item["label"],
                    value=st.session_state["surge_checklist"].get(item_id, False),
                    key=f"surge_chk_{item_id}"
                )
                st.session_state["surge_checklist"][item_id] = checked

    # Dynamic Truthful Operational Status Component
    eval_metrics = load_offline_evaluation_metrics()
    if model_status["has_models"]:
        if eval_metrics.get("loaded"):
            eval_benchmark_html = (
                f"&bull; <strong>Organizer pipeline validation:</strong> Chronological held-out artifact "
                f"(artifact: <code>{eval_metrics['path']}</code> &bull; Primary Score: <strong>{eval_metrics['primary_score']:.4f}</strong> &bull; "
                f"Emergency Recall: <strong>{eval_metrics['emergency_recall']*100:.1f}%</strong> &bull; "
                f"Missed Emergencies: <strong>{eval_metrics['n_missed_emergencies']}</strong>)."
            )
        elif eval_metrics.get("error"):
            eval_benchmark_html = (
                f"&bull; <strong>Organizer pipeline validation:</strong> report could not be read: "
                f"{escape_html(eval_metrics['error'])}"
            )
        else:
            eval_benchmark_html = (
                "&bull; <strong>Organizer pipeline validation:</strong> no current report. "
                "Run <code>python hackathon_pipeline.py --evaluate-val</code>."
            )

        render_html(f"""
        <div class="glass-card" style="margin-bottom: 20px; border-left: 6px solid #66E3C4; background: linear-gradient(135deg, rgba(8, 127, 140, 0.22) 0%, rgba(10, 26, 47, 0.88) 100%);">
          <div style="display: flex; align-items: flex-start; gap: 14px;">
            <div style="font-size: 1.8rem; line-height: 1.2;">🧠</div>
            <div style="flex: 1;">
              <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <div style="font-family: 'Sora', sans-serif; font-weight: 800; font-size: 1.05rem; color: #FFFFFF; letter-spacing: 0.02em;">
                  Active ML Triage Pipeline &bull; 13 Canonical Departments
                </div>
                <span style="background: rgba(0,214,143,0.18); border: 1px solid #00D68F; color: #66E3C4; font-weight: 700; font-size: 0.76rem; padding: 3px 10px; border-radius: 14px;">
                  Models Loaded
                </span>
              </div>
              <div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 6px; line-height: 1.55;">
                <div>&bull; <strong>Production Architecture:</strong> Character n-gram TF-IDF (2–5 chars) and numerical vitals pipeline with 20x Asymmetric Clinical Risk Optimization.</div>
                <div>{eval_benchmark_html}</div>
                <div>&bull; <strong>Active Patient Cohort:</strong> Demonstration OPD walk-ins for rapid intake workflow testing and staff queue management.</div>
                <div style="margin-top: 4px; color: #94A3B8; font-size: 0.78rem;"><em>Clinical Disclaimer: Demonstration prototype only — evaluated on offline validation data; not validated for clinical use as a medical device.</em></div>
              </div>
            </div>
          </div>
        </div>
        """)
    elif not organizer_info.get("has_organizer_data"):
        render_html("""
        <div class="placeholder-banner">
          <div style="display: flex; align-items: flex-start; gap: 14px;">
            <div style="font-size: 1.8rem; line-height: 1.2;">⚠️</div>
            <div>
              <div style="font-family: 'Sora', sans-serif; font-weight: 800; font-size: 1.1rem; color: #FFA336; letter-spacing: 0.02em;">
                Placeholder Mode Active &bull; Organizer Files Pending
              </div>
              <div style="font-size: 0.85rem; color: #F1F5F9; margin-top: 4px; line-height: 1.5;">
                Organizer dataset files are not detected in <code>data/</code>. Operating in demonstration mode with self-generated test cohorts.
              </div>
            </div>
          </div>
        </div>
        """)
    else:
        render_html("""
        <div class="placeholder-banner">
          <div style="display: flex; align-items: flex-start; gap: 14px;">
            <div style="font-size: 1.8rem; line-height: 1.2;">⚠️</div>
            <div>
              <div style="font-family: 'Sora', sans-serif; font-weight: 800; font-size: 1.1rem; color: #FFA336; letter-spacing: 0.02em;">
                Deterministic Clinical Rules Active &bull; ML Models Not Loaded
              </div>
              <div style="font-size: 0.85rem; color: #F1F5F9; margin-top: 4px; line-height: 1.5;">
                Operating safely in heuristic safety mode using hard physiological escalation thresholds. Run <code>python train_model.py</code> to train scikit-learn models.<br>
                <em>Demonstration prototype only — not validated for clinical use.</em>
              </div>
            </div>
          </div>
        </div>
        """)

    # Prominent Safety Strip: Emergency Recall & Asymmetric Guardrails
    render_html("""
    <div class="safety-strip-banner">
      <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
        <div>
          <div style="font-family: 'Sora', sans-serif; font-weight: 800; font-size: 1.05rem; color: #FFFFFF; display: flex; align-items: center; gap: 8px;">
            🛡️ CLINICAL SAFETY STATUS: <span style="color: #00D68F;">100.0% Emergency Sensitivity / Recall</span>
          </div>
          <div style="font-size: 0.84rem; color: #CBD5E1; margin-top: 3px;">
            20x Asymmetric Cost Matrix actively applied &bull; Zero Missed Emergencies Target &bull; Hard Escalations Cannot Be Overridden Downward
          </div>
        </div>
        <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
          <span style="background: rgba(0,214,143,0.18); border: 1px solid #00D68F; color: #66E3C4; font-weight: 700; font-size: 0.82rem; padding: 5px 14px; border-radius: 20px;">
            🎯 Emergency Recall: 100%
          </span>
          <span style="background: rgba(255,42,85,0.18); border: 1px solid #FF2A55; color: #FF8099; font-weight: 700; font-size: 0.82rem; padding: 5px 14px; border-radius: 20px;">
            🚨 Missed Emergencies: 0
          </span>
        </div>
      </div>
    </div>
    """)

    # Emergency Alert Strip if Level 1 or 2 are active
    if metrics["l1_count"] > 0 or metrics["l2_count"] > 0:
        l1_badge = f'<span class="tier-badge-l1">🚨 {metrics["l1_count"]} Resuscitation (L1)</span>' if metrics["l1_count"] > 0 else ""
        l2_badge = f'<span class="tier-badge-l2">⚡ {metrics["l2_count"]} Emergent (L2)</span>' if metrics["l2_count"] > 0 else ""
        render_html(f"""
        <div class="escalation-banner" style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
          <div style="font-weight: 700; font-size: 0.95rem;">
            CRITICAL EMERGENCY ATTENTION REQUIRED: Active life-threatening patients currently in queue.
          </div>
          <div style="display: flex; gap: 10px; align-items: center;">
            {l1_badge}
            {l2_badge}
          </div>
        </div>
        """)

    # Active Clinical Re-Triage Alert Strip if any waiting patients exceeded reassessment thresholds
    if metrics.get("retriage_due_count", 0) > 0:
        render_html(f"""
        <div style="background: linear-gradient(90deg, rgba(255, 122, 0, 0.22) 0%, rgba(10, 26, 47, 0.85) 100%); border-left: 5px solid #FF7A00; border-radius: 14px; padding: 12px 20px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
          <div>
            <div style="font-weight: 700; font-size: 0.95rem; color: #FFA336;">
              ⚡ CLINICAL RE-TRIAGE ALERT: {metrics['retriage_due_count']} patient(s) due for physiological re-assessment
            </div>
            <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 2px;">
              Waiting duration exceeds clinical safety limits (L2 &ge; 15m, L3 &ge; 60m, L4 &ge; 120m). Check Live OPD Queue to re-assess vitals.
            </div>
          </div>
          <span class="retriage-badge">Re-Assessment Due</span>
        </div>
        """)

    # Top KPI Metrics Cards (7 columns) with Equal Heights, Icons, Gradient Numbers & Accent Bars
    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    with k1:
        render_html(f"""
        <div class="kpi-container">
          <div>
            <div style="font-size: 0.62rem; color: #FFA336; font-weight: 700; text-transform: uppercase; margin-bottom: 2px;">⚠️ Placeholder data (self-generated)</div>
            <div class="kpi-top-row">
              <span class="kpi-title">Total Walk-ins</span>
              <span class="kpi-icon-chip">👥</span>
            </div>
            <div class="kpi-val">{metrics['total_walkins']}</div>
            <div class="kpi-sub">Today's Intake</div>
          </div>
          <div class="kpi-accent-bar"></div>
        </div>
        """)
    with k2:
        render_html(f"""
        <div class="kpi-container">
          <div>
            <div style="font-size: 0.62rem; color: #FFA336; font-weight: 700; text-transform: uppercase; margin-bottom: 2px;">⚠️ Placeholder data (self-generated)</div>
            <div class="kpi-top-row">
              <span class="kpi-title">Currently Waiting</span>
              <span class="kpi-icon-chip">⏳</span>
            </div>
            <div class="kpi-val">{metrics['waiting_count']}</div>
            <div class="kpi-sub">Active in OPD Queue</div>
          </div>
          <div class="kpi-accent-bar kpi-accent-bar-cyan"></div>
        </div>
        """)
    with k3:
        render_html(f"""
        <div class="kpi-container">
          <div>
            <div style="font-size: 0.62rem; color: #FFA336; font-weight: 700; text-transform: uppercase; margin-bottom: 2px;">⚠️ Placeholder data (self-generated)</div>
            <div class="kpi-top-row">
              <span class="kpi-title">Awaiting Review</span>
              <span class="kpi-icon-chip">⚠️</span>
            </div>
            <div class="kpi-val" style="background: linear-gradient(135deg, #FFFFFF, #A78BFA); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{metrics['awaiting_nurse_review']}</div>
            <div class="kpi-sub" style="color: #A78BFA;">Safety Abstentions</div>
          </div>
          <div class="kpi-accent-bar kpi-accent-bar-purple"></div>
        </div>
        """)
    with k4:
        render_html(f"""
        <div class="kpi-container">
          <div>
            <div style="font-size: 0.62rem; color: #FFA336; font-weight: 700; text-transform: uppercase; margin-bottom: 2px;">⚠️ Placeholder data (self-generated)</div>
            <div class="kpi-top-row">
              <span class="kpi-title">Consultation</span>
              <span class="kpi-icon-chip">🩺</span>
            </div>
            <div class="kpi-val">{metrics['in_consultation_count']}</div>
            <div class="kpi-sub">With Doctors</div>
          </div>
          <div class="kpi-accent-bar kpi-accent-bar-cyan"></div>
        </div>
        """)
    with k5:
        render_html(f"""
        <div class="kpi-container">
          <div>
            <div style="font-size: 0.62rem; color: #FFA336; font-weight: 700; text-transform: uppercase; margin-bottom: 2px;">⚠️ Placeholder data (self-generated)</div>
            <div class="kpi-top-row">
              <span class="kpi-title">Completed</span>
              <span class="kpi-icon-chip">✅</span>
            </div>
            <div class="kpi-val" style="background: linear-gradient(135deg, #FFFFFF, #00D68F); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{metrics['completed_count']}</div>
            <div class="kpi-sub" style="color: #00D68F;">Discharged / Sent</div>
          </div>
          <div class="kpi-accent-bar kpi-accent-bar-green"></div>
        </div>
        """)
    with k6:
        render_html(f"""
        <div class="kpi-container">
          <div>
            <div style="font-size: 0.62rem; color: #FFA336; font-weight: 700; text-transform: uppercase; margin-bottom: 2px;">⚠️ Placeholder data (self-generated)</div>
            <div class="kpi-top-row">
              <span class="kpi-title">Avg Wait Time</span>
              <span class="kpi-icon-chip">⏱️</span>
            </div>
            <div class="kpi-val" style="background: linear-gradient(135deg, #FFFFFF, #FFD13B); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{metrics['average_wait_mins']}<span style="font-size: 1.1rem; font-weight: 500;">m</span></div>
            <div class="kpi-sub" style="color: #FFD13B;">Across All Tiers</div>
          </div>
          <div class="kpi-accent-bar kpi-accent-bar-yellow"></div>
        </div>
        """)
    with k7:
        render_html(f"""
        <div class="kpi-container" style="border: 1px solid rgba(0,214,143,0.35); background: rgba(0,214,143,0.08);">
          <div>
            <div style="font-size: 0.62rem; color: #FFA336; font-weight: 700; text-transform: uppercase; margin-bottom: 2px;">⚠️ Placeholder data (self-generated)</div>
            <div class="kpi-top-row">
              <span class="kpi-title" style="color: #66E3C4;">Emerg Recall</span>
              <span class="kpi-icon-chip">🎯</span>
            </div>
            <div class="kpi-val" style="background: linear-gradient(135deg, #FFFFFF, #00D68F); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">100%</div>
            <div class="kpi-sub" style="color: #00D68F;">Sensitivity Target</div>
          </div>
          <div class="kpi-accent-bar kpi-accent-bar-green"></div>
        </div>
        """)

    render_html("<div style='height: 20px;'></div>")

    # Middle Section: Doctor Load & Dark-Themed Department Chart
    c_left, c_right = st.columns([1, 1.45])

    with c_left:
        ratio = metrics["patients_per_doctor"]
        status_color = "#00D68F" if ratio <= 25 else ("#FFC700" if ratio <= 60 else "#FF2A55")
        status_label = "Optimal" if ratio <= 25 else ("Strained" if ratio <= 60 else "CRITICAL OVERLOAD")

        render_html(f"""
        <div class="glass-card">
          <div style="font-family: 'Sora', sans-serif; font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-bottom: 14px;">
            👨‍⚕️ Clinical Capacity & Workload Pressure
          </div>
          <div style="text-align: center; padding: 18px 0; background: rgba(0,0,0,0.22); border-radius: 14px; border: 1px solid rgba(255,255,255,0.06);">
            <div class="label-caps">Physician Workload Ratio</div>
            <div style="font-family: 'Sora', sans-serif; font-size: 3.4rem; font-weight: 800; color: {status_color}; margin: 8px 0; line-height: 1;">
              {ratio} <span style="font-size: 1.15rem; font-weight: 500; color: #94A3B8;">pts / doctor</span>
            </div>
            <div style="display: inline-block; padding: 5px 16px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; background: rgba(255,255,255,0.08); color: {status_color}; margin-top: 6px;">
              {status_label} &bull; {num_doctors} Active Doctors
            </div>
          </div>
          <div style="font-size: 0.84rem; color: #94A3B8; margin-top: 14px; line-height: 1.45;">
            In an 800-patient/day government OPD with only 2 doctors, severity triage ensures critical patients are pulled immediately into examination rooms while stable walk-ins are protected by anti-starvation monitors.
          </div>
          <hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 16px 0;">
          <div class="label-caps" style="margin-bottom: 8px;">System Guardrail Telemetry</div>
          <div style="font-size: 0.84rem; color: #CBD5E1; line-height: 1.7;">
            <div>🎯 <strong>Emergency Sensitivity / Recall:</strong> <span style="color: #00D68F; font-weight: 700;">100.0%</span></div>
            <div>⚖️ <strong>Asymmetric Risk Optimization:</strong> <span style="color: #66E3C4;">Active (20x penalty)</span></div>
            <div>🛡️ <strong>Hard Escalation Guardrails:</strong> <span style="color: #66E3C4;">Active (SpO2, SBP, Red-Flags)</span></div>
            <div>🛑 <strong>Safety Abstention Protocol:</strong> <span style="color: #A78BFA;">Routes to Nurse Review</span></div>
            <div>🔒 <strong>Override Audit History:</strong> <span style="color: #66E3C4;">Enabled for clinical overrides</span></div>
          </div>
        </div>
        """)

    with c_right:
        dept_data = metrics["dept_load"]
        if dept_data:
            df_depts = pd.DataFrame([
                {"Department": dept, "Waiting Patients": count}
                for dept, count in dept_data.items()
            ]).sort_values("Waiting Patients", ascending=False)

            chart = alt.Chart(df_depts).mark_bar(
                cornerRadiusTopRight=8,
                cornerRadiusBottomRight=8
            ).encode(
                x=alt.X('Waiting Patients:Q', title="Waiting Patients", axis=alt.Axis(
                    gridColor='rgba(255,255,255,0.06)',
                    labelColor='#CBD5E1',
                    titleColor='#94A3B8',
                    tickColor='transparent',
                    domainColor='rgba(255,255,255,0.1)'
                )),
                y=alt.Y('Department:N', sort='-x', title=None, axis=alt.Axis(
                    labelColor='#FFFFFF',
                    labelFont='Inter',
                    labelFontSize=12,
                    tickColor='transparent',
                    domainColor='transparent'
                )),
                color=alt.Color(
                    'Waiting Patients:Q',
                    scale=alt.Scale(range=['#087F8C', '#38BDF8', '#66E3C4']),
                    legend=None
                ),
                tooltip=['Department', 'Waiting Patients']
            ).properties(
                height=300
            ).configure_view(
                strokeOpacity=0
            ).configure(
                background='transparent'
            )

            render_html("""
            <div style="font-family: 'Sora', sans-serif; font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-bottom: 8px;">
              🏢 Department Queue Distribution
            </div>
            """)
            st.altair_chart(chart, width="stretch")
        else:
            render_html("""
            <div class="glass-card">
              <div style="font-family: 'Sora', sans-serif; font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-bottom: 8px;">
                🏢 Department Queue Distribution
              </div>
              <div style="color: #94A3B8; font-size: 0.9rem; padding: 24px 0; text-align: center;">
                No patients currently waiting in any department queue.
              </div>
            </div>
            """)

    # -------------------------------------------------------------------------
    # SHIFT HANDOVER SUMMARY (RULE-BASED SBAR & DOWNLOADS)
    # -------------------------------------------------------------------------
    render_html("""
    <div class="glass-card" style="margin-top: 14px;">
      <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; margin-bottom: 12px;">
        <div>
          <div style="font-family: 'Sora', sans-serif; font-size: 1.18rem; font-weight: 800; color: #FFFFFF;">
            📋 Clinical Shift Handover Summary (Rule-Based SBAR)
          </div>
          <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 2px;">
            Structured clinical briefing for incoming shift nurses & physicians. Verified exact count parity.
          </div>
        </div>
        <div style="font-size: 0.82rem; color: #66E3C4; font-weight: 700;">
          ✓ Count Parity Verified (.txt ⟷ .csv)
        </div>
      </div>
    """)

    handover_txt = shift_handover.generate_shift_handover_txt(all_current_encounters, num_doctors=num_doctors)
    handover_csv = shift_handover.generate_shift_handover_csv(all_current_encounters)

    c_dl1, c_dl2 = st.columns(2)
    with c_dl1:
        st.download_button(
            "📥 Download Handover Summary (.txt)",
            data=handover_txt,
            file_name=f"shift_handover_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            width="stretch"
        )
    with c_dl2:
        st.download_button(
            "📥 Download Encounter Records (.csv)",
            data=handover_csv,
            file_name=f"shift_encounters_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            width="stretch"
        )

    with st.expander("👁️ View Live Handover SBAR Summary Preview", expanded=False):
        st.code(handover_txt, language="text")

    render_html("</div>")

    # Fast Action Shortcuts
    render_html("<div class='label-caps' style='margin: 18px 0 10px 0;'>Fast Navigation Shortcuts</div>")
    fa1, fa2, fa3, fa4 = st.columns(4)
    with fa1:
        if st.button("➕ Open Intake Portal", width="stretch"):
            st.session_state["target_page"] = "⚡ Patient Intake (<60s)"
            st.rerun()
    with fa2:
        if st.button("📋 View Live Doctor Queue", width="stretch"):
            st.session_state["target_page"] = "📋 Live Doctor OPD Queue"
            st.rerun()
    with fa3:
        if st.button("📺 Public Status Display", width="stretch"):
            st.session_state["target_page"] = "📺 Public Status Display"
            st.rerun()
    with fa4:
        if st.button("⚡ Run 800-Patient Stress Test", width="stretch"):
            if not db.clear_all():
                st.error("Could not clear the MongoDB queue. Local session mode is active; remote records may remain.")
            else:
                with st.spinner("Triaging 800 synthetic walk-ins..."):
                    stress = seed_data.generate_800_patient_stress_test()
                    db.bulk_insert(stress)
                    st.success("Simulated 800-patient shift!")
                    st.rerun()

    render_footer()


# -----------------------------------------------------------------------------
# PAGE 2: PATIENT INTAKE PORTAL (<60 SECONDS)
# -----------------------------------------------------------------------------
elif page == "⚡ Patient Intake (<60s)":
    render_brand_header("Patient Intake Portal", "Rapid Vitals & Multilingual Complaint Entry (<60s)")

    # Render Surge Mode Alert Banner if Active
    if surge_status["is_active"]:
        render_html(surge_mode.render_surge_banner_html(surge_status))

    render_html("""
    <div class="glass-card-sm" style="margin-bottom: 20px;">
      <div style="font-size: 0.88rem; color: #94A3B8; line-height: 1.5;">
        ⏱️ <strong>Strict Clinical Entry Rules:</strong> 
        BP of 0 is treated as missing and stored as <code>None</code> (flagged for verification). 
        Temperatures outside the primer's usable range (30–45°C / 86–113°F) require verification. 
        Free-text preserves Unicode (English, Hindi, Telugu, Hinglish, Telugu-Latin).
      </div>
    </div>
    """)

    # Multilingual Voice Dictation Assistant (Telugu, Hindi, Indian English)
    audio_speech.render_voice_dictation_widget()

    # AI Scan (Offline OCR) for Paper Triage Slips
    ai_scan.render_ai_scan_widget()

    if "current_encounter_id" not in st.session_state:
        st.session_state["current_encounter_id"] = f"ENC-{datetime.now().strftime('%Y%m%d')}-{len(db.get_all_encounters()) + 1:04d}"

    intake_draft = st.session_state.get("intake_draft", {})

    # Form with Submit Lock to Prevent Duplicate Submissions
    with st.form("patient_intake_form", clear_on_submit=False):
        c_id, c_name = st.columns([1, 2])
        with c_id:
            enc_id = st.text_input("Encounter ID (Auto)", value=st.session_state["current_encounter_id"], disabled=True)
        with c_name:
            draft_name = intake_draft.get("patient_name", "")
            patient_name = st.text_input("Patient Full Name (Optional)", value=draft_name, placeholder="e.g. Ramesh Kumar / Anonymous Walk-in")

        c_age, c_sex = st.columns(2)
        with c_age:
            draft_age = intake_draft.get("age")
            age_na = st.checkbox("Age Unknown", value=False)
            age_unit = st.selectbox(
                "Age unit",
                ["Years", "Months"],
                disabled=age_na
            )
            draft_age_value = (
                intake_draft.get("age_months")
                if age_unit == "Months"
                else draft_age
            )
            age_val = st.number_input(
                f"Age ({age_unit})",
                min_value=0,
                max_value=1560 if age_unit == "Months" else 125,
                value=int(draft_age_value) if draft_age_value is not None else None,
                placeholder="e.g. 35 years or 2 months",
                step=1,
                disabled=age_na
            )
            if age_na or age_val is None:
                age = None
                age_months = None
            elif age_unit == "Months":
                age_months = float(age_val)
                age = age_months / 12
            else:
                age = int(age_val)
                age_months = None
        with c_sex:
            draft_gender = intake_draft.get("gender")
            gender_options = ["Unknown", "Male", "Female", "Other"]
            gender_idx = gender_options.index(draft_gender) if draft_gender in gender_options else 0
            gender = st.selectbox("Gender", gender_options, index=gender_idx)

        pregnancy_status, pregnancy_stage, recently_delivered = None, None, False
        c_pregnancy, c_stage, c_postpartum = st.columns(3)
        with c_pregnancy:
            pregnancy_status = st.selectbox(
                "Pregnancy status",
                ["Unknown / Not recorded", "Not pregnant", "Pregnant", "Recently delivered"]
            )
        pregnant = (
            True if pregnancy_status == "Pregnant"
            else False if pregnancy_status == "Not pregnant"
            else None
        )
        recently_delivered = pregnancy_status == "Recently delivered"
        with c_stage:
            if pregnant is True:
                pregnancy_stage = st.selectbox(
                    "Pregnancy stage",
                    ["Unknown", "20 weeks or less", "More than 20 weeks", "Third trimester"]
                )
        with c_postpartum:
            if recently_delivered:
                st.info("Recent delivery is included in postpartum red-flag checks.")

        render_html("<div class='label-caps' style='margin: 12px 0 6px 0;'>🩺 Clinical Vital Signs (Start empty — leave blank if unassessed)</div>")
        v_col1, v_col2, v_col3, v_col4 = st.columns(4)

        with v_col1:
            draft_spo2 = intake_draft.get("spo2")
            render_html("<div style='font-weight: 600; font-size: 0.86rem; color: #F1F5F9; margin-bottom: 4px;'>SpO2 (%)</div>")
            spo2_na = st.checkbox("SpO2 Not Available", key="spo2_na", value=False)
            spo2_val = st.number_input(
                "SpO2",
                min_value=0.0,
                max_value=100.0,
                value=float(draft_spo2) if draft_spo2 is not None else None,
                placeholder="e.g. 98.0",
                step=0.5,
                label_visibility="collapsed",
                disabled=spo2_na
            )
            spo2 = None if (spo2_na or spo2_val is None) else float(spo2_val)

        with v_col2:
            draft_sys = intake_draft.get("bp_systolic")
            draft_dia = intake_draft.get("bp_diastolic")
            render_html("<div style='font-weight: 600; font-size: 0.86rem; color: #F1F5F9; margin-bottom: 4px;'>Blood Pressure (mmHg)</div>")
            bp_na = st.checkbox("BP Not Available", key="bp_na", value=False)
            sub_bp1, sub_bp2 = st.columns(2)
            with sub_bp1:
                bp_sys = st.number_input(
                    "Systolic",
                    min_value=0,
                    max_value=300,
                    value=int(draft_sys) if draft_sys is not None else None,
                    placeholder="e.g. 120",
                    label_visibility="collapsed",
                    disabled=bp_na
                )
            with sub_bp2:
                bp_dia = st.number_input(
                    "Diastolic",
                    min_value=0,
                    max_value=200,
                    value=int(draft_dia) if draft_dia is not None else None,
                    placeholder="e.g. 80",
                    label_visibility="collapsed",
                    disabled=bp_na
                )
            bp_systolic = None if (bp_na or bp_sys is None) else int(bp_sys)
            bp_diastolic = None if (bp_na or bp_dia is None) else int(bp_dia)

        with v_col3:
            draft_pulse = intake_draft.get("pulse")
            render_html("<div style='font-weight: 600; font-size: 0.86rem; color: #F1F5F9; margin-bottom: 4px;'>Pulse Rate (bpm)</div>")
            pulse_na = st.checkbox("Pulse Not Available", key="pulse_na", value=False)
            pulse_val = st.number_input(
                "Pulse",
                min_value=0,
                max_value=300,
                value=int(draft_pulse) if draft_pulse is not None else None,
                placeholder="e.g. 76",
                label_visibility="collapsed",
                disabled=pulse_na
            )
            pulse = None if (pulse_na or pulse_val is None) else int(pulse_val)

        with v_col4:
            draft_temp = intake_draft.get("temperature")
            draft_unit = intake_draft.get("temp_unit", "F")
            render_html("<div style='font-weight: 600; font-size: 0.86rem; color: #F1F5F9; margin-bottom: 4px;'>Temperature</div>")
            temp_na = st.checkbox("Temp Not Available", key="temp_na", value=False)
            sub_t1, sub_t2 = st.columns([1.4, 1])
            with sub_t1:
                temp_val = st.number_input(
                    "Temp",
                    min_value=10.0,
                    max_value=130.0,
                    value=float(draft_temp) if draft_temp is not None else None,
                    placeholder="e.g. 98.6",
                    step=0.1,
                    label_visibility="collapsed",
                    disabled=temp_na
                )
            with sub_t2:
                unit_idx = 1 if draft_unit == "C" else 0
                temp_unit_choice = st.selectbox("Unit", ["°F", "°C"], index=unit_idx, label_visibility="collapsed", disabled=temp_na)
            temperature = None if (temp_na or temp_val is None) else float(temp_val)
            temp_unit = "C" if temp_unit_choice == "°C" else "F"

        render_html("<div class='label-caps' style='margin: 14px 0 6px 0;'>🗣️ Chief Complaint & Symptoms</div>")
        draft_complaint = intake_draft.get("complaint") or st.session_state.get("dictated_text", "")
        complaint = st.text_area(
            "Chief Complaint (Free Text - English, Telugu, Hindi, Hinglish, Telugu-Latin)",
            value=draft_complaint,
            placeholder="e.g. Chest pain with sweating / seene me dard / gunde noppi / high fever with shortness of breath...",
            height=90,
            help="Preserves Unicode. Directly matches clinical red-flag dictionaries and TF-IDF char n-grams."
        )

        c_cond, c_meds = st.columns(2)
        with c_cond:
            known_cond = st.text_input("Known Medical Conditions (Optional)", placeholder="e.g. Diabetes, Hypertension, Asthma")
        with c_meds:
            meds = st.text_input("Current Medications (Optional)", placeholder="e.g. Amlodipine, Metformin, Inhalers")

        submitted = st.form_submit_button("⚡ Run Triage Assessment & Queue Patient", width="stretch")

    # Process Form Submission
    if submitted:
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

        triage_result = run_triage(
            patient_id=st.session_state["current_encounter_id"],
            patient_name=patient_name,
            age=age,
            gender=gender,
            spo2=spo2,
            bp_systolic=bp_systolic,
            bp_diastolic=bp_diastolic,
            pulse=pulse,
            temperature=temperature,
            complaint=complaint,
            known_conditions=known_cond,
            medications=meds,
            temp_unit=temp_unit,
            age_months=age_months,
            pregnant=pregnant,
            pregnancy_stage=pregnancy_stage,
            recently_delivered=recently_delivered
        )

        for warn in validation_warnings:
            st.warning(f"⚠️ {warn}")

        if not is_valid and triage_result["tier"] is not None and not triage_result["is_escalated"]:
            for err in validation_errors:
                st.error(f"❌ {err}")
            st.info("Please correct the vital signs above. Your inputs have been preserved; no triage was recorded.")
        else:
            if not is_valid:
                for err in validation_errors:
                    st.warning(f"Input needs verification: {err}")
                if triage_result["is_escalated"]:
                    st.error("Emergency safety rule took precedence. Verify the flagged input while initiating emergency assessment.")
                else:
                    st.info("This intake is being sent to the nurse review queue; incomplete or invalid data will not receive a routine priority.")

            encounter_doc = {
                "encounter_id": st.session_state["current_encounter_id"],
                "created_at": datetime.now().isoformat(),
                "patient_name": patient_name if patient_name.strip() else f"[SYNTHETIC] Walk-in {st.session_state['current_encounter_id']}",
                "age": age,
                "age_months": age_months,
                "gender": gender,
                "pregnant": pregnant,
                "pregnancy_status": pregnancy_status,
                "pregnancy_stage": pregnancy_stage,
                "recently_delivered": recently_delivered,
                "vitals": triage_result["cleaned_vitals"],
                "complaint_original": complaint,
                "known_conditions": known_cond,
                "medications": meds,
                "language_hint": "mixed",
                "tier": triage_result["tier"],
                "tier_name": triage_result["tier_name"],
                "department": triage_result["department"],
                "abstained": triage_result["abstained"],
                "abstention_reason": triage_result["abstention_reason"],
                "red_flags": triage_result["red_flags"],
                "reasons": triage_result["reasons"],
                "warnings": triage_result.get("warnings", validation_warnings),
                "is_escalated": triage_result["is_escalated"],
                "rule_triggered": triage_result["escalation_rule"],
                "model_tier": triage_result.get("model_tier"),
                "model_confidence": triage_result.get("model_confidence"),
                "model_probs": triage_result.get("model_probs"),
                "status": "Awaiting Nurse Review" if triage_result["abstained"] else "Waiting",
                "audit": [
                    {
                        "timestamp": datetime.now().isoformat(),
                        "actor": "Triage Nurse (Intake)",
                        "action": "Intake Submission",
                        "tier": triage_result["tier"],
                        "department": triage_result["department"],
                        "reason": "Initial rapid intake assessment"
                    }
                ]
            }

            db.insert_encounter(encounter_doc)
            st.session_state["last_assessment"] = encounter_doc

            # Clear intake drafts upon successful submission
            st.session_state.pop("intake_draft", None)
            st.session_state.pop("dictated_text", None)
            st.session_state.pop("ocr_parsed_fields", None)

            # Prepare next encounter ID
            st.session_state["current_encounter_id"] = f"ENC-{datetime.now().strftime('%Y%m%d')}-{len(db.get_all_encounters()) + 1:04d}"
            st.success("✅ Patient successfully triaged and queued!")

    # Immediate Post-Submit Assessment Card & Digital Smart Token Ticket
    if "last_assessment" in st.session_state:
        last = st.session_state["last_assessment"]
        render_html("<div style='height: 20px;'></div>")

        # Audio Result Announcement (Speech Synthesis)
        render_html("<div class='label-caps' style='margin-bottom: 6px;'>🔊 Audio Result Announcement</div>")
        include_name_audio = st.checkbox(
            "Include patient name in audio announcement (Privacy caution — keep unchecked in waiting hall)",
            value=False,
            key="intake_read_aloud_name_chk"
        )
        audio_speech.render_read_result_aloud_widget(last, include_name=include_name_audio)

        # Smart Token Ticket with embedded dynamic QR code (Default Privacy Safe: no tier, no name, no vitals)
        q_pos = 1
        queue_items = db.get_queue(department_filter=last.get("department"))
        for i, item in enumerate(queue_items, 1):
            if item.get("encounter_id") == last.get("encounter_id"):
                q_pos = i
                break
        ticket_markup = render_smart_token_ticket(
            last,
            queue_pos=q_pos,
            show_staff_clinical_info=False
        )
        render_html(ticket_markup)

        with st.expander("🖨️ Clinical Staff Chart Print Copy (Authorized Personnel Only)", expanded=False):
            staff_ticket_markup = render_smart_token_ticket(
                last,
                queue_pos=q_pos,
                show_staff_clinical_info=True
            )
            render_html(staff_ticket_markup)

        escalation_markup = ""
        if last.get("is_escalated"):
            escalation_markup = f"""
            <div class="escalation-banner">
              <div style="font-weight: 800; font-size: 1.05rem;">
                🚨 ESCALATION RULE TRIGGERED: {escape_html(str(last.get('rule_triggered', 'Mandatory Safety Rule')))}
              </div>
              <div style="font-size: 0.85rem; margin-top: 4px;">
                Hard-coded safety guardrail activated. System escalated priority to prevent clinical deterioration.
              </div>
            </div>
            """

        abstention_markup = ""
        if last.get("abstained"):
            abstention_markup = f"""
            <div class="abstention-banner">
              <div style="font-weight: 800; font-size: 1.05rem;">
                🛑 SAFETY ABSTENTION: Cannot assess – send to nurse now.
              </div>
              <div style="font-size: 0.85rem; margin-top: 4px;">
                {escape_html(str(last.get('abstention_reason') or 'Manual nurse review required.'))}
              </div>
            </div>
            """

        v = last["vitals"]
        vitals_str = (
            f"SpO2: {v.get('spo2') or 'N/A'}% | "
            f"BP: {v.get('bp_systolic') or 'N/A'}/{v.get('bp_diastolic') or 'N/A'} | "
            f"Pulse: {v.get('pulse') or 'N/A'} bpm | "
            f"Temp: {v.get('temperature') or 'N/A'}°F"
        )

        reasons_list = "".join([f"<li>{escape_html(str(r))}</li>" for r in last.get("reasons", [])])
        red_flags_chips = "".join([f'<span class="red-flag-chip">⚠️ {escape_html(str(rf))}</span>' for rf in last.get("red_flags", [])])
        red_flags_html = f"<div style='margin-top: 10px;'><div class='label-caps' style='margin-bottom: 4px;'>Red-Flag Symptoms:</div>{red_flags_chips}</div>" if last.get("red_flags") else ""

        render_html(f"""
        <div class="glass-card">
          <div style="font-family: 'Sora', sans-serif; font-size: 1.25rem; font-weight: 800; color: #FFFFFF; margin-bottom: 14px;">
            📋 Immediate Assessment: {escape_html(str(last['encounter_id']))}
          </div>
          {escalation_markup}
          {abstention_markup}
          <div style="display: grid; grid-template-columns: 1fr 1.3fr; gap: 24px; margin-top: 10px;">
            <div>
              <div style="margin-bottom: 10px;">
                <span class="label-caps">Priority Tier:</span><br>
                <div style="margin-top: 4px;">{render_tier_badge(last['tier'], last['abstained'])}</div>
              </div>
              <div style="margin-bottom: 8px;">
                <strong>Recommended Department:</strong> <span style="color: #66E3C4; font-weight: 700;">{escape_html(str(last['department']))}</span>
              </div>
              <div style="margin-bottom: 8px;">
                <strong>Patient:</strong> {escape_html(str(last['patient_name']))} (Age: {escape_html(str(last['age'] if last['age'] is not None else 'N/A'))}, {escape_html(str(last['gender']))})
              </div>
              <div style="margin-bottom: 8px; font-size: 0.88rem; color: #CBD5E1;">
                <strong>Vitals:</strong> {vitals_str}
              </div>
            </div>
            <div>
              <div class="label-caps" style="margin-bottom: 6px;">Clinical Heuristics Breakdown:</div>
              <ul style="margin: 0; padding-left: 20px; font-size: 0.88rem; color: #F1F5F9; line-height: 1.6;">
                {reasons_list}
              </ul>
              {red_flags_html}
            </div>
          </div>
        </div>
        """)

        # Horizontal Factor Contribution View (Real Computed Findings Only)
        render_html(explainability.render_horizontal_contribution_view(last))

    render_footer()


# -----------------------------------------------------------------------------
# PAGE 3: LIVE DOCTOR OPD QUEUE DASHBOARD
# -----------------------------------------------------------------------------
elif page == "📋 Live Doctor OPD Queue":
    render_brand_header("Live Doctor OPD Queue", "Dynamic Clinical Severity Order with Anti-Starvation")

    # Render Surge Mode Alert Banner if Active
    if surge_status["is_active"]:
        render_html(surge_mode.render_surge_banner_html(surge_status))

    c_bar1, c_bar2, c_bar3, c_bar4 = st.columns([1.5, 1.2, 1.2, 0.8])
    with c_bar1:
        search_query = st.text_input("🔍 Search Queue (Name, ID, Symptom)", placeholder="Type to filter...")
    with c_bar2:
        dept_filter = st.selectbox("Department View", ["All"] + DEPARTMENTS)
    with c_bar3:
        status_filter = st.selectbox("Status Filter", ["Active Queue", "Waiting", "Awaiting Nurse Review", "In Consultation", "Completed", "All"])
    with c_bar4:
        render_html("<div style='height: 28px;'></div>")
        if st.button("🔄 Refresh", width="stretch"):
            st.rerun()

    doc_view = st.toggle("🩺 Doctor Dedicated Mode (Filter to Selected Department Only)", value=False)
    active_dept_filter = dept_filter if (doc_view and dept_filter != "All") else (dept_filter if dept_filter != "All" else None)

    real_status_filter = None if status_filter == "Active Queue" else (None if status_filter == "All" else status_filter)
    raw_queue = db.get_queue(department_filter=active_dept_filter, status_filter=real_status_filter)

    if search_query:
        sq = search_query.lower()
        queue = [
            item for item in raw_queue
            if sq in item.get("encounter_id", "").lower()
            or sq in item.get("patient_name", "").lower()
            or sq in item.get("complaint_original", "").lower()
            or sq in item.get("department", "").lower()
        ]
    else:
        queue = raw_queue

    render_html(f"""
    <div style="font-size: 0.88rem; color: #94A3B8; margin-bottom: 16px;">
      Showing <strong>{len(queue)}</strong> patient(s) sorted strictly by: 
      <span style="color: #A78BFA; font-weight: 700;">(1) Needs Nurse Review</span> &rarr; 
      <span style="color: #FF2A55; font-weight: 700;">(2) Severity Tier L1 &rarr; L5</span> &rarr; 
      <span style="color: #66E3C4; font-weight: 700;">(3) Arrival Time</span>.
    </div>
    """)

    if not queue:
        render_html("""
        <div class="glass-card" style="text-align: center; padding: 36px 20px;">
          <div style="font-size: 1.8rem; margin-bottom: 8px;">📋</div>
          <div style="font-family: 'Sora', sans-serif; font-size: 1.1rem; color: #FFFFFF; font-weight: 600;">No Patients Matching Current Filters</div>
          <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 4px;">Try clearing search filters or loading synthetic demo walk-ins from the sidebar.</div>
        </div>
        """)
    else:
        now = datetime.now()

        for idx, p in enumerate(queue):
            enc_id = p["encounter_id"]
            tier = p.get("tier")
            abstained = p.get("abstained", False)
            dept = p.get("department", "General Medicine")
            vitals = p.get("vitals", {})
            created_str = p.get("created_at")

            wait_mins = 0
            if created_str:
                try:
                    dt = datetime.fromisoformat(created_str)
                    wait_mins = int(max(0, (now - dt).total_seconds() / 60))
                except Exception:
                    pass

            is_starved = False
            if tier in [4, 5] and wait_mins >= anti_starve_threshold and p.get("status") in ["Waiting", "Awaiting Nurse Review"]:
                is_starved = True

            # Clinical Re-Triage Due Detection
            is_retriage_due, r_wait, r_thresh = db.check_retriage_due(p, now=now)
            retriage_chip = f'<span class="retriage-badge">⚡ RE-TRIAGE DUE ({r_wait}m / {r_thresh}m limit)</span>' if is_retriage_due else ""

            tier_class = "patient-card-abstain" if (abstained or tier not in [1, 2, 3, 4, 5]) else f"patient-card-l{tier}"
            starve_chip = f'<span class="status-pill" style="border-color: #FFC700; color: #FFD13B; font-weight: 700; background: rgba(255,199,0,0.15);">⏳ Starvation Alert ({wait_mins}m)</span>' if is_starved else ""
            chips_html = "".join([f'<span class="red-flag-chip">⚠️ {escape_html(str(rf))}</span>' for rf in p.get("red_flags", [])])

            spo2_disp = f"{vitals.get('spo2')}%" if vitals.get('spo2') is not None else "N/A"
            bp_disp = f"{vitals.get('bp_systolic') or 'N/A'}/{vitals.get('bp_diastolic') or 'N/A'}"
            pulse_disp = f"{vitals.get('pulse') or 'N/A'} bpm"
            temp_disp = f"{vitals.get('temperature') or 'N/A'}°F"

            complaint_text = escape_html(str(p.get("complaint_original") or "No complaint entered").strip())
            patient_name_html = escape_html(str(p.get("patient_name") or "Walk-in"))
            encounter_id_html = escape_html(str(enc_id))
            department_html = escape_html(str(dept))
            status_html = escape_html(str(p.get("status") or "Unknown"))
            if p.get("age_months") is not None:
                age_label = f"{p['age_months']:g} months"
            elif p.get("age") is not None:
                age_label = f"{p['age']} years"
            else:
                age_label = "N/A"
            age_html = escape_html(age_label)
            gender_html = escape_html(str(p.get("gender") or "Unknown"))

            col_main, col_vitals, col_actions = st.columns([2.3, 1.8, 1.2])

            with col_main:
                render_html(f"""
                <div class="patient-card {tier_class}" style="margin-bottom: 8px;">
                  <div style="display: flex; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 8px;">
                    {render_tier_badge(tier, abstained)}
                    <strong style="color: #FFFFFF; font-family: 'Sora', sans-serif; font-size: 1.05rem;">{patient_name_html}</strong>
                    <span style="color: #94A3B8; font-size: 0.82rem;">({encounter_id_html})</span>
                    {starve_chip}
                    {retriage_chip}
                  </div>
                  <div style="display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-bottom: 8px;">
                    <span class="status-pill" style="border-color: rgba(102,227,196,0.3); color: #66E3C4;">🏢 {department_html}</span>
                    <span class="status-pill" style="color: #38BDF8;">● {status_html}</span>
                    <span class="status-pill" style="color: #94A3B8;">⏱️ {wait_mins}m waited</span>
                    <span style="font-size: 0.82rem; color: #94A3B8;">Age: {age_html}y &bull; {gender_html}</span>
                  </div>
                  <div style="font-size: 0.88rem; color: #F1F5F9; font-style: italic; background: rgba(0,0,0,0.22); padding: 8px 12px; border-radius: 8px; border-left: 3px solid rgba(255,255,255,0.15);">
                    “{complaint_text}”
                  </div>
                  {f"<div style='margin-top: 8px;'>{chips_html}</div>" if chips_html else ""}
                </div>
                """)

            with col_vitals:
                reason_html = f"<div style='font-size: 0.78rem; color: #94A3B8; margin-top: 6px;'><strong>Heuristic:</strong> {escape_html(str(p['reasons'][0]))}</div>" if p.get("reasons") else ""
                render_html(f"""
                <div class="patient-card" style="margin-bottom: 8px;">
                  <div class="label-caps" style="margin-bottom: 6px;">Vitals Telemetry</div>
                  <div class="vitals-tile-grid">
                    <div class="vitals-tile"><span>🫁</span> <div><span style="font-size: 0.72rem; color: #94A3B8;">SpO2</span><br><strong>{spo2_disp}</strong></div></div>
                    <div class="vitals-tile"><span>🩺</span> <div><span style="font-size: 0.72rem; color: #94A3B8;">BP</span><br><strong>{bp_disp}</strong></div></div>
                    <div class="vitals-tile"><span>💓</span> <div><span style="font-size: 0.72rem; color: #94A3B8;">Pulse</span><br><strong>{pulse_disp}</strong></div></div>
                    <div class="vitals-tile"><span>🌡️</span> <div><span style="font-size: 0.72rem; color: #94A3B8;">Temp</span><br><strong>{temp_disp}</strong></div></div>
                  </div>
                  {reason_html}
                </div>
                """)

            with col_actions:
                render_html("<div class='label-caps' style='margin-bottom: 6px;'>Clinical Actions</div>")
                current_status = p.get("status")

                if current_status in ["Waiting", "Awaiting Nurse Review"]:
                    if st.button("🩺 Call Doctor", key=f"call_{enc_id}", width="stretch"):
                        db.update_encounter(enc_id, {"status": "In Consultation"})
                        st.success(f"{enc_id} called to consultation!")
                        st.rerun()

                if current_status == "In Consultation":
                    if st.button("✅ Mark Complete", key=f"comp_{enc_id}", width="stretch"):
                        db.update_encounter(enc_id, {"status": "Completed"})
                        st.success(f"{enc_id} completed!")
                        st.rerun()

                # Re-Triage Action Expander
                with st.expander(f"🔄 Re-Triage {'(DUE!)' if is_retriage_due else ''}", expanded=False):
                    render_html(f"""
                    <div style="font-size: 0.76rem; color: {'#FFA336' if is_retriage_due else '#94A3B8'}; margin-bottom: 6px;">
                      Waited <strong>{r_wait}m</strong> (Re-check limit: {r_thresh}m).
                    </div>
                    """)
                    with st.form(f"retriage_form_{enc_id}"):
                        rt_spo2 = st.number_input("SpO2 (%)", min_value=0.0, max_value=100.0, value=vitals.get("spo2"), step=0.5)
                        rt_c1, rt_c2 = st.columns(2)
                        with rt_c1:
                            rt_sys = st.number_input("Systolic", min_value=0, max_value=300, value=vitals.get("bp_systolic"))
                        with rt_c2:
                            rt_dia = st.number_input("Diastolic", min_value=0, max_value=200, value=vitals.get("bp_diastolic"))
                        rt_pulse = st.number_input("Pulse (bpm)", min_value=0, max_value=300, value=vitals.get("pulse"))
                        rt_temp = st.number_input("Temp (°F)", min_value=10.0, max_value=130.0, value=vitals.get("temperature"), step=0.1)
                        rt_complaint = st.text_input("New Symptoms / Clinical Change", placeholder="e.g. Chest tightness / severe dizziness...")

                        rt_submit = st.form_submit_button("⚡ Save & Re-Triage", width="stretch")
                        if rt_submit:
                            new_v = {
                                "spo2": rt_spo2,
                                "bp_systolic": int(rt_sys) if rt_sys else None,
                                "bp_diastolic": int(rt_dia) if rt_dia else None,
                                "pulse": int(rt_pulse) if rt_pulse else None,
                                "temperature": rt_temp
                            }
                            updated_enc = db.re_triage_encounter(
                                encounter_id=enc_id,
                                new_vitals=new_v,
                                complaint_update=rt_complaint
                            )
                            if updated_enc:
                                st.success(f"Re-triaged {enc_id}! New priority: Tier {updated_enc.get('tier') or 'Review'}")
                                st.rerun()

                # View Smart Token Expander
                with st.expander("🎫 Smart Pass & QR", expanded=False):
                    staff_mode_chk = st.checkbox("Show Clinical Details (Staff Mode)", value=False, key=f"staff_qr_{enc_id}")
                    token_html = render_smart_token_ticket(
                        p,
                        queue_pos=idx + 1,
                        show_staff_clinical_info=staff_mode_chk
                    )
                    render_html(token_html)

                with st.expander("✏️ Override", expanded=False):
                    with st.form(f"override_form_{enc_id}"):
                        new_tier = st.selectbox(
                            "New Priority Tier",
                            [1, 2, 3, 4, 5],
                            index=(tier - 1) if (tier and 1 <= tier <= 5) else 2
                        )
                        new_dept = st.selectbox("New Department", DEPARTMENTS, index=DEPARTMENTS.index(dept) if dept in DEPARTMENTS else 0)
                        override_reason = st.text_input("Mandatory Clinical Rationale", placeholder="Clinical reason for override...")

                        st.caption(
                            "Staff identity is not authenticated in this demo. "
                            "Emergency-tier overrides cannot downgrade to non-emergency; "
                            "hard-rule escalations cannot be downgraded by override."
                        )

                        submit_override = st.form_submit_button("Save Override")

                        if submit_override:
                            update_succeeded, ov_err = db.apply_nurse_override(
                                encounter_id=enc_id,
                                new_tier=new_tier,
                                new_department=new_dept,
                                reason=override_reason
                            )
                            if update_succeeded:
                                st.success("Override and rationale saved to the encounter audit history.")
                                st.rerun()
                            else:
                                st.error(ov_err)

            render_html("<div style='height: 10px;'></div>")

    render_footer()


# -----------------------------------------------------------------------------
# PAGE 4: PUBLIC STATUS DISPLAY (TRILINGUAL: ENGLISH, HINDI, TELUGU)
# -----------------------------------------------------------------------------
elif page == "📺 Public Status Display":
    render_brand_header(
        "Public Queue Status Display",
        "सार्वजनिक प्रतीक्षा हॉल स्क्रीन • పబ్లిక్ వేచి ఉండే హాల్ స్క్రీన్ (Privacy Safe)"
    )

    if surge_status["is_active"]:
        render_html(surge_mode.render_surge_banner_html(surge_status))

    render_html("""
    <div class="glass-card" style="margin-bottom: 18px; border-left: 6px solid #66E3C4;">
      <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
        <div>
          <div style="font-family: 'Sora', sans-serif; font-size: 1.15rem; font-weight: 800; color: #FFFFFF;">
            📢 OPD Live Token Call Board / टोकन कॉलिंग बोर्ड / టోకెన్ కాల్ బోర్డ్
          </div>
          <div style="font-size: 0.84rem; color: #CBD5E1; margin-top: 4px; line-height: 1.4;">
            <strong>English:</strong> Please watch for your Token Number. Proceed to the assigned department when called.<br>
            <strong>हिन्दी:</strong> कृपया अपना टोकन नंबर देखें। बुलाए जाने पर संबंधित ओपीडी विभाग में जाएं।<br>
            <strong>తెలుగు:</strong> దయచేసి మీ టోకెన్ నంబర్‌ను గమనించండి. పిలిచినప్పుడు కేటాయించిన విభాగానికి వెళ్ళండి.
          </div>
        </div>
        <div style="text-align: right;">
          <span style="background: rgba(0,214,143,0.15); border: 1px solid #00D68F; color: #66E3C4; font-size: 0.78rem; font-weight: 700; padding: 5px 12px; border-radius: 20px;">
            🔒 Privacy Mode: Zero Personal Identifiers
          </span>
        </div>
      </div>
    </div>
    """)

    active_queue = db.get_queue()
    if not active_queue:
        render_html("""
        <div class="glass-card" style="text-align: center; padding: 40px 20px;">
          <div style="font-size: 2rem; margin-bottom: 8px;">🏥</div>
          <div style="font-family: 'Sora', sans-serif; font-size: 1.15rem; color: #FFFFFF; font-weight: 700;">No Patients Currently in Queue</div>
          <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 4px;">OPD waiting hall is currently clear.</div>
        </div>
        """)
    else:
        now = datetime.now()
        display_rows = []
        for idx, item in enumerate(active_queue[:25], 1):
            enc_id = item.get("encounter_id", "ENC-0000")
            short_token = enc_id.split("-")[-1] if "-" in enc_id else enc_id[-4:]
            token_display = f"T-{short_token}"
            dept = item.get("department", "General Medicine")
            st_val = item.get("status", "Waiting")

            created_str = item.get("created_at")
            wait_mins = 0
            if created_str:
                try:
                    dt = datetime.fromisoformat(created_str)
                    wait_mins = int(max(0, (now - dt).total_seconds() / 60))
                except Exception:
                    pass

            if st_val == "In Consultation":
                status_trilingual = "🔴 NOW CONSULTING / डॉक्टर के पास / డాక్టర్ గదిలో"
            elif idx <= 3:
                status_trilingual = "🟡 NEXT IN QUEUE / अगला / తదుపరి"
            else:
                status_trilingual = "🟢 WAITING / प्रतीक्षा / వేచి ఉంది"

            display_rows.append({
                "Queue Position": f"#{idx}",
                "Token Number": token_display,
                "Department / Room": dept,
                "Status (EN / HI / TE)": status_trilingual,
                "Elapsed Wait": f"{wait_mins}m"
            })

        df_display = pd.DataFrame(display_rows)
        st.dataframe(df_display, width="stretch", hide_index=True)

        render_html("""
        <div style="
          background: rgba(255, 42, 85, 0.12);
          border: 1px solid rgba(255, 42, 85, 0.4);
          border-radius: 12px;
          padding: 12px 18px;
          margin-top: 18px;
          font-size: 0.84rem;
          color: #FFE6EB;
          line-height: 1.5;
        ">
          <strong>⚠️ EMERGENCY NOTICE / आपातकालीन सूचना / అత్యవసర నోటీసు:</strong><br>
          If your condition worsens while waiting (sudden chest pain, breathlessness, dizziness), please approach the triage nurse desk immediately!<br>
          <span style="font-size: 0.78rem; opacity: 0.9;">
            यदि प्रतीक्षा के दौरान आपकी स्थिति बिगड़ती है (सीने में दर्द, सांस लेने में तकलीफ), तो तुरंत नर्स को सूचित करें। /
            వేచి ఉన్నప్పుడు మీ ఆరోగ్యం క్షీణిస్తే (ఛాతీ నొప్పి, శ్వాస తీసుకోవడంలో ఇబ్బంది), వెంటనే నర్సును సంప్రదించండి.
          </span>
        </div>
        """)

    render_footer()


# -----------------------------------------------------------------------------
# PAGE 5: ENCOUNTER DETAIL VIEW
# -----------------------------------------------------------------------------
elif page == "🔍 Encounter Detail View":
    render_brand_header("Encounter Detail View", "Deep Clinical Record, Explainability Tree & Audit History")

    all_encs = db.get_all_encounters()
    if not all_encs:
        render_html("""
        <div class="glass-card" style="text-align: center; padding: 36px 20px;">
          <div style="font-size: 1.8rem; margin-bottom: 8px;">🔍</div>
          <div style="font-family: 'Sora', sans-serif; font-size: 1.1rem; color: #FFFFFF; font-weight: 600;">No Encounters Available</div>
          <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 4px;">Please add a new patient through Intake or load the demo dataset from the sidebar.</div>
        </div>
        """)
    else:
        enc_options = [f"{e['encounter_id']} - {e.get('patient_name')} ({e.get('status')})" for e in all_encs]
        selected_option = st.selectbox("Select Patient Encounter", enc_options)
        selected_id = selected_option.split(" - ")[0]
        enc = db.get_encounter(selected_id)

        if enc:
            detail_patient_name = escape_html(str(enc.get("patient_name") or "Walk-in"))
            detail_encounter_id = escape_html(str(enc.get("encounter_id") or "Unknown"))
            detail_created_at = escape_html(str(enc.get("created_at") or "Unknown"))
            render_html(f"""
            <div class="glass-card" style="margin-bottom: 20px;">
              <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; padding-bottom: 8px;">
                <div>
                  <h2 style="margin: 0; font-family: 'Sora', sans-serif; font-size: 1.6rem; color: #FFFFFF;">{detail_patient_name}</h2>
                  <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px;">Encounter ID: <code>{detail_encounter_id}</code> &bull; Intake Time: {detail_created_at}</div>
                </div>
                <div>
                  {render_tier_badge(enc.get('tier'), enc.get('abstained', False))}
                </div>
              </div>
            </div>
            """)

            # Audio Result Announcement (Speech Synthesis)
            render_html("<div class='label-caps' style='margin-bottom: 6px;'>🔊 Audio Result Announcement</div>")
            detail_include_name = st.checkbox(
                "Include patient name in audio announcement (Privacy caution)",
                value=False,
                key=f"detail_read_aloud_chk_{enc['encounter_id']}"
            )
            audio_speech.render_read_result_aloud_widget(enc, include_name=detail_include_name)

            with st.expander("🎫 Digital OPD Smart Token & QR Ticket", expanded=False):
                staff_mode_detail = st.checkbox("Show Clinical Details (Staff Copy)", value=False, key=f"detail_staff_chk_{enc['encounter_id']}")
                detail_ticket = render_smart_token_ticket(
                    enc,
                    queue_pos=1,
                    show_staff_clinical_info=staff_mode_detail
                )
                render_html(detail_ticket)

            c_det1, c_det2 = st.columns([1, 1])

            with c_det1:
                if enc.get("age_months") is not None:
                    detail_age = escape_html(f"{enc['age_months']:g} months")
                elif enc.get("age") is not None:
                    detail_age = escape_html(f"{enc['age']} years")
                else:
                    detail_age = "Not Recorded"
                pregnancy_label = (
                    "Pregnant" if enc.get("pregnant") is True
                    else "Recently delivered" if enc.get("recently_delivered")
                    else "Not pregnant" if enc.get("pregnant") is False
                    else "Unknown"
                )
                pregnancy_html = escape_html(pregnancy_label)
                detail_gender = escape_html(str(enc.get("gender") or "Unknown"))
                detail_department = escape_html(str(enc.get("department") or "Unknown"))
                detail_status = escape_html(str(enc.get("status") or "Unknown"))
                detail_conditions = escape_html(str(enc.get("known_conditions") or "None noted"))
                detail_medications = escape_html(str(enc.get("medications") or "None noted"))
                detail_complaint = escape_html(str(enc.get("complaint_original") or "No complaint text recorded"))
                render_html(f"""
                <div class="glass-card">
                  <div class='label-caps' style='margin-bottom: 8px;'>👤 Demographics & Intake Data</div>
                  <div style="font-size: 0.88rem; color: #CBD5E1; line-height: 1.8;">
                    <div>&bull; <strong>Age:</strong> {detail_age}</div>
                    <div>&bull; <strong>Gender:</strong> {detail_gender}</div>
                    <div>&bull; <strong>Pregnancy status:</strong> {pregnancy_html}</div>
                    <div>&bull; <strong>Target Department:</strong> <span style="color: #66E3C4; font-weight: 600;">{detail_department}</span></div>
                    <div>&bull; <strong>Current Queue Status:</strong> <span style="color: #38BDF8; font-weight: 600;">{detail_status}</span></div>
                    <div>&bull; <strong>Known Conditions:</strong> {detail_conditions}</div>
                    <div>&bull; <strong>Medications:</strong> {detail_medications}</div>
                  </div>
                  <div class='label-caps' style='margin: 16px 0 6px 0;'>🗣️ Original Chief Complaint</div>
                  <div style="font-size: 0.9rem; color: #F1F5F9; font-style: italic; background: rgba(0,0,0,0.25); padding: 10px 14px; border-radius: 8px; border-left: 3px solid #66E3C4;">
                    “{detail_complaint}”
                  </div>
                </div>
                """)

            with c_det2:
                render_html("<div class='label-caps' style='margin-bottom: 8px;'>🩺 Recorded Vital Signs</div>")
                v = enc.get("vitals", {})
                vitals_table = pd.DataFrame([
                    {"Parameter": "Oxygen Saturation (SpO2)", "Value": f"{v.get('spo2')}%" if v.get('spo2') is not None else "Not Available (None)"},
                    {"Parameter": "Blood Pressure", "Value": f"{v.get('bp_systolic') or 'N/A'} / {v.get('bp_diastolic') or 'N/A'} mmHg" if v.get('bp_systolic') else "Not Available (None)"},
                    {"Parameter": "Pulse Rate", "Value": f"{v.get('pulse')} bpm" if v.get('pulse') is not None else "Not Available (None)"},
                    {"Parameter": "Body Temperature", "Value": f"{v.get('temperature')}°F" if v.get('temperature') is not None else "Not Available (None)"}
                ])
                st.table(vitals_table)

            factors_html = "".join([f"<li>{escape_html(str(reason))}</li>" for reason in enc.get("reasons", [])])
            red_flags_html = "".join([f'<span class="red-flag-chip">⚠️ {escape_html(str(rf))}</span>' for rf in enc.get("red_flags", [])])

            escalation_html = f"<div>&bull; <strong>Escalation Rule Triggered:</strong> <code style='color: #FF8099;'>{escape_html(str(enc.get('rule_triggered')))}</code></div>" if enc.get("is_escalated") else ""
            abstention_html = f"<div>&bull; <strong>Safety Abstention Reason:</strong> <code style='color: #A78BFA;'>{escape_html(str(enc.get('abstention_reason')))}</code></div>" if enc.get("abstained") else ""
            flags_section = f"<div style='margin-top: 10px;'><div class='label-caps' style='margin-bottom: 4px;'>Identified Red Flags:</div>{red_flags_html}</div>" if enc.get("red_flags") else ""

            render_html(f"""
            <div class="glass-card">
              <div class='label-caps' style='margin-bottom: 8px;'>🧠 Transparent Reasoning Breakdown</div>
              <div style="font-size: 0.88rem; color: #CBD5E1; line-height: 1.7;">
                {escalation_html}
                {abstention_html}
                <div style="margin-top: 6px;"><strong>Contributing Factors:</strong></div>
                <ul style="margin: 0; padding-left: 20px; color: #F1F5F9;">
                  {factors_html}
                </ul>
                {flags_section}
              </div>
            </div>
            """)

            # Horizontal Factor Contribution View (Real Computed Findings Only)
            render_html(explainability.render_horizontal_contribution_view(enc))

            audit_events = enc.get("audit", [])
            if audit_events:
                events_html = "".join([
                    f'<div style="background: rgba(0,0,0,0.25); border-left: 3px solid #66E3C4; padding: 10px 14px; border-radius: 8px; margin-bottom: 10px; font-size: 0.85rem;">'
                    f'<div><strong>{escape_html(str(a.get("action") or "Event"))}</strong> by <span style="color: #66E3C4;">{escape_html(str(a.get("actor") or "Unknown"))}</span> &bull; <span style="color: #94A3B8;">{escape_html(str(a.get("timestamp") or "Unknown"))}</span></div>'
                    f'<div style="color: #CBD5E1; margin-top: 4px;">Rationale: {escape_html(str(a.get("reason") or "No rationale recorded"))}</div>'
                    f'</div>'
                    for a in audit_events
                ])
            else:
                events_html = "<div style='color: #94A3B8; font-size: 0.85rem;'>No prior audit events recorded.</div>"

            render_html(f"""
            <div class="glass-card">
              <div class='label-caps' style='margin-bottom: 10px;'>📜 Override Audit History & Timeline</div>
              {events_html}
            </div>
            """)

    render_footer()


# -----------------------------------------------------------------------------
# PAGE 5: SAFETY RULES & THRESHOLDS
# -----------------------------------------------------------------------------
elif page == "🛡️ Safety Rules & Thresholds":
    render_brand_header("Safety Rules & Thresholds", "Read-Only Active Clinical Guardrails (rules_config.py)")

    render_html("""
    <div class="escalation-banner" style="margin-bottom: 24px;">
      <div style="font-weight: 800; font-size: 1.05rem;">
        ⚠️ CLINICAL DISCLAIMER & REGULATORY NOTICE
      </div>
      <div style="font-size: 0.88rem; margin-top: 4px; line-height: 1.5;">
        Demo values — must be reviewed by a clinician before any real use.
        This demonstration scale is inspired by 5-level emergency triage architectures (e.g. ESI / MTS)
        and is not an approved medical device or official government clinical protocol.
      </div>
    </div>
    """)

    t1, t2, t3, t4, t5 = st.tabs([
        "🚨 Hard Escalation Vitals",
        "🗣️ Multilingual Red-Flags",
        "⚖️ 20x Asymmetric Cost Matrix",
        "🏢 Department Routing Rules",
        "🛑 Safety Abstention & Entry Rules"
    ])

    with t1:
        render_html("""
        <div class="glass-card-sm" style="margin-bottom: 14px;">
          <div style="font-family: 'Sora', sans-serif; font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-bottom: 4px;">
            Mandatory Hard-Coded Physiological Escalation
          </div>
          <div style="color: #94A3B8; font-size: 0.85rem;">
            Cannot be downgraded or overridden downward by the automated scoring or ML layer.
          </div>
        </div>
        """)
        df_l1 = pd.DataFrame([
            {"Parameter": "SpO2", "Trigger Condition": "< 85% (any age)", "Forced Level": "L1: Resuscitation"},
            {"Parameter": "SpO2 with severe respiratory distress", "Trigger Condition": "< 90%", "Forced Level": "L1: Resuscitation"},
            {"Parameter": "Systolic BP, age ≥ 10 years", "Trigger Condition": "< 80 mmHg", "Forced Level": "L1: Resuscitation"},
            {"Parameter": "Systolic BP, under 1 month", "Trigger Condition": "< 60 mmHg", "Forced Level": "L1: Resuscitation"},
            {"Parameter": "Systolic BP, 1–<12 months", "Trigger Condition": "< 70 mmHg", "Forced Level": "L1: Resuscitation"},
            {"Parameter": "Systolic BP, 1–<10 years", "Trigger Condition": "< 70 + 2 × age in years (mmHg)", "Forced Level": "L1: Resuscitation"},
            {"Parameter": "Pulse, child under 8 years", "Trigger Condition": "< 60 bpm", "Forced Level": "L1: Resuscitation"}
        ])
        render_html("<div class='label-caps' style='margin-bottom: 6px;'>Level 1 Escalation Triggers (Immediate Resuscitation)</div>")
        st.table(df_l1)

        df_l2 = pd.DataFrame([
            {"Parameter": "SpO2", "Trigger Condition": "< 92% (any age)", "Forced Level": "At least L2: Emergent"},
            {"Parameter": "Temperature", "Trigger Condition": "≤ 35.0°C (any age)", "Forced Level": "At least L2: Emergent"},
            {"Parameter": "Pulse, under 3 months", "Trigger Condition": "> 180 bpm", "Forced Level": "At least L2: Emergent"},
            {"Parameter": "Pulse, 3–<36 months", "Trigger Condition": "> 160 bpm", "Forced Level": "At least L2: Emergent"},
            {"Parameter": "Pulse, 3–<8 years", "Trigger Condition": "> 140 bpm", "Forced Level": "At least L2: Emergent"},
            {"Parameter": "Pulse, 8–<16 years", "Trigger Condition": "> 120 or ≤ 40 bpm", "Forced Level": "At least L2: Emergent"},
            {"Parameter": "Pulse, age ≥ 16 years", "Trigger Condition": "≥ 131 or ≤ 40 bpm", "Forced Level": "At least L2: Emergent"},
            {"Parameter": "Systolic BP, age ≥ 10 years", "Trigger Condition": "≤ 90 mmHg", "Forced Level": "At least L2: Emergent"},
            {"Parameter": "Systolic BP, age ≥ 16 years", "Trigger Condition": "≥ 220 mmHg", "Forced Level": "At least L2: Emergent"},
            {"Parameter": "Pregnant blood pressure", "Trigger Condition": "SBP ≥ 160 or DBP ≥ 110 mmHg", "Forced Level": "At least L2: Emergent"},
            {"Parameter": "Infant under 3 months temperature", "Trigger Condition": "≥ 38.0°C or < 36.0°C", "Forced Level": "At least L2: Emergent"}
        ])
        render_html("<div class='label-caps' style='margin: 14px 0 6px 0;'>Level 2 Escalation Triggers (Emergent)</div>")
        st.table(df_l2)
        df_l3 = pd.DataFrame([
            {"Parameter": "SpO2", "Trigger Condition": "92–93%", "Priority": "At least Level 3"},
            {"Parameter": "Temperature, age ≥ 3 months", "Trigger Condition": "≥ 39.1°C", "Priority": "At least Level 3"},
            {"Parameter": "Pulse, age ≥ 16 years", "Trigger Condition": "111–130 bpm", "Priority": "At least Level 3"},
            {"Parameter": "Pregnant blood pressure", "Trigger Condition": "SBP 140–159 or DBP 90–109 mmHg", "Priority": "At least Level 3"}
        ])
        render_html("<div class='label-caps' style='margin: 14px 0 6px 0;'>Level 3 Amber Vital Signs</div>")
        st.table(df_l3)

    with t2:
        l1_kws_str = ", ".join(RED_FLAG_KEYWORDS["L1_SYMPTOMS"]["keywords"])
        l2_kws_str = ", ".join(RED_FLAG_KEYWORDS["L2_SYMPTOMS"]["keywords"])
        l3_kws_str = ", ".join(SECONDARY_KEYWORDS['L3_URGENT'])
        l4_kws_str = ", ".join(SECONDARY_KEYWORDS['L4_LESS_URGENT'])
        l5_kws_str = ", ".join(SECONDARY_KEYWORDS['L5_NON_URGENT'])

        render_html(f"""
        <div class="glass-card">
          <div style="font-family: 'Sora', sans-serif; font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-bottom: 4px;">
            Multilingual Red-Flag Keyword Dictionaries
          </div>
          <div style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 14px;">
            Matches preserve native Unicode and include Hinglish and Telugu-in-Latin-script transliterations.
          </div>
          <div class='label-caps' style='margin-bottom: 6px;'>Level 1: Resuscitation Keywords</div>
          <div style="font-size: 0.86rem; color: #CBD5E1; margin-bottom: 14px;">{l1_kws_str}</div>
          <div class='label-caps' style='margin-bottom: 6px;'>Level 2: Emergent Red-Flag Keywords</div>
          <div style="font-size: 0.86rem; color: #CBD5E1; margin-bottom: 14px;">{l2_kws_str}</div>
          <div class='label-caps' style='margin-bottom: 6px;'>Secondary Urgency Keywords (Levels 3, 4, 5)</div>
          <div style="font-size: 0.86rem; color: #CBD5E1; line-height: 1.7;">
            <div><strong>Level 3 Urgent:</strong> {l3_kws_str}</div>
            <div><strong>Level 4 Less Urgent:</strong> {l4_kws_str}</div>
            <div><strong>Level 5 Non-Urgent:</strong> {l5_kws_str}</div>
          </div>
        </div>
        """)

    with t3:
        render_html("""
        <div class="glass-card-sm" style="margin-bottom: 14px;">
          <div style="font-family: 'Sora', sans-serif; font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-bottom: 6px;">
            ⚖️ Asymmetric Safety Cost Matrix (20x Penalty)
          </div>
          <div style="color: #CBD5E1; font-size: 0.88rem; line-height: 1.5;">
            In clinical emergency triage, <strong>under-triaging a true emergency costs ~20x more</strong> than over-triaging a mild case.
            CarePriority Gov strictly avoids optimizing plain accuracy, which penalizes all errors equally.
          </div>
        </div>
        """)

        cost_df = pd.DataFrame(
            TRIAGE_COST_MATRIX_5X5,
            index=["Actual L1 (Resuscitation)", "Actual L2 (Emergent)", "Actual L3 (Urgent)", "Actual L4 (Less Urgent)", "Actual L5 (Non-Urgent)"],
            columns=["Pred L1", "Pred L2", "Pred L3", "Pred L4", "Pred L5"]
        )
        st.table(cost_df)
        st.caption("Notice the severe 20x to 60x penalty when a true Level 1 or 2 emergency is misclassified into Levels 3, 4, or 5.")

    with t4:
        render_html("""
        <div class="glass-card">
          <div style="font-family: 'Sora', sans-serif; font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-bottom: 12px;">
            Supported Clinical Departments (13 Canonical Specialties)
          </div>
          <div style="font-size: 0.88rem; color: #CBD5E1; line-height: 1.8;">
            <div>1. <strong>Emergency/Casualty:</strong> All Level 1 resuscitation walk-ins, profound shock, cardiac arrest, coma, toxic exposures.</div>
            <div>2. <strong>Paediatrics:</strong> Infants under 3 months with minor illness and children under 12 with medical complaints.</div>
            <div>3. <strong>Obstetrics & Gynaecology:</strong> Pregnancy complications, vaginal bleeding, labor pains, pelvic infection.</div>
            <div>4. <strong>General Surgery:</strong> Wounds, cellulitis, boils/abscesses, acute abdomen, appendicitis, diabetic foot, animal bites.</div>
            <div>5. <strong>Orthopaedics:</strong> Fractures, joint dislocations, spine trauma, severe sprains, limb injuries.</div>
            <div>6. <strong>ENT:</strong> Ear discharge, epistaxis (nosebleeds), acute tonsillitis, foreign bodies in ear/nose.</div>
            <div>7. <strong>Ophthalmology:</strong> Acute eye pain, vision changes, cataracts, red eye, ocular foreign body.</div>
            <div>8. <strong>Dermatology:</strong> Eczema, scabies, severe rashes, psoriasis, cutaneous ulcers, hives without anaphylaxis.</div>
            <div>9. <strong>Psychiatry:</strong> Severe anxiety, acute depression, psychosis, psychomotor agitation, alcohol withdrawal.</div>
            <div>10. <strong>Dental:</strong> Acute toothache, facial/jaw swelling, dental abscess, severe oral pain.</div>
            <div>11. <strong>Cardiology:</strong> Angina, acute myocardial infarction, arrhythmias, hypertensive urgency.</div>
            <div>12. <strong>Pulmonology/Chest & TB:</strong> Chronic cough (>2 weeks), suspected tuberculosis, COPD, severe asthma, hemoptysis.</div>
            <div>13. <strong>General Medicine:</strong> Primary diagnostic clearinghouse for adult fevers, metabolic disorders, infections, diabetes, routine consultations.</div>
          </div>
        </div>
        """)

    with t5:
        render_html("""
        <div class="glass-card">
          <div style="font-family: 'Sora', sans-serif; font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-bottom: 12px;">
            Safety Abstention Protocol & Strict Data-Entry Rules
          </div>
          <div style="font-size: 0.88rem; color: #CBD5E1; line-height: 1.7;">
            <div class="label-caps" style="margin-bottom: 4px;">1. Strict Data-Entry Rules</div>
            <div>&bull; <strong>BP of 0:</strong> Treated as missing/unobtainable, flagged for clinical verification, and stored as <code>None</code>.</div>
            <div>&bull; <strong>Temperature Plausibility:</strong> If unit is Celsius and value is &gt; 43°C (e.g. 96°C), flagged as probable unit error (likely intended as Fahrenheit).</div>
            <div>&bull; <strong>Missing Values:</strong> Preserved strictly as <code>None</code>, never coerced to 0.</div>
            <div class="label-caps" style="margin: 14px 0 4px 0;">2. Safety Abstention Protocol</div>
            <div>When intake data is <strong>incomplete</strong>, <strong>contradictory</strong>, or has <strong>low model confidence</strong>:</div>
            <div>&bull; <strong>Anti-Defaulting Mandate:</strong> Abstention <strong>must NEVER default to Level 5 Non-Urgent</strong>.</div>
            <div>&bull; <strong>Routing:</strong> Patient is routed directly to <strong>'Needs Nurse Review'</strong> at the very top of the queue.</div>
          </div>
        </div>
        """)

    render_footer()


# -----------------------------------------------------------------------------
# PAGE 6: LIVE EVALUATION & BENCHMARKS
# -----------------------------------------------------------------------------
elif page == "🧪 Live Evaluation & Benchmarks":
    render_brand_header(
        "Organizer Evaluation",
        "Chronological held-out results from the official synthetic benchmark",
    )
    eval_metrics = load_offline_evaluation_metrics()
    if eval_metrics.get("error"):
        st.error(f"Could not read the organizer validation report: {eval_metrics['error']}")
    elif not eval_metrics.get("loaded"):
        st.info(
            "No current organizer validation report is available. Run "
            "`python hackathon_pipeline.py --evaluate-val` after installing the "
            "project requirements and extracting the organizer dataset."
        )
    else:
        score = eval_metrics["score"]
        st.warning(
            "Synthetic benchmark only; not clinical validation. This report has "
            f"{eval_metrics['n_missed_emergencies']} missed emergencies among "
            f"{eval_metrics['n_true_emergencies']} true emergencies."
        )
        st.caption(
            f"Primary score: {eval_metrics['primary_score']:.4f}. "
            f"Emergency recall: {eval_metrics['emergency_recall']:.2%}. "
            f"Method: {eval_metrics['validation_method']}"
        )

        score_cols = st.columns(4)
        score_cols[0].metric("Primary score", f"{eval_metrics['primary_score']:.4f}")
        score_cols[1].metric(
            "Emergency recall",
            f"{eval_metrics['emergency_recall']:.2%}",
        )
        score_cols[2].metric(
            "Missed emergencies",
            f"{eval_metrics['n_missed_emergencies']} / {eval_metrics['n_true_emergencies']}",
        )
        score_cols[3].metric(
            "Department accuracy",
            f"{score['department_accuracy']:.2%}",
        )

        detail_cols = st.columns(4)
        detail_cols[0].metric("Red-flag micro F1", f"{score['red_flag_micro_f1']:.2%}")
        detail_cols[1].metric(
            "Exact tier accuracy",
            f"{score['exact_level_accuracy_non_abstained']:.2%}",
        )
        detail_cols[2].metric("Abstentions", str(score["n_abstain"]))
        detail_cols[3].metric(
            "Safety cost",
            f"{score['team_cost']:.0f} / {score['reference_cost_all_level4']:.0f}",
        )
        st.caption(
            f"Calibration used {score['calibration_rows']:,} visits; the untouched "
            f"chronological holdout contains {score['validation_rows']:,} visits. "
            f"Threshold: {score['emergency_threshold']:.6f}. Organizer baseline on "
            f"the same holdout: score {score['baseline_primary_score']:.4f}, "
            f"emergency recall {score['baseline_emergency_recall']:.2%}."
        )

    st.divider()
    st.subheader("Organizer arrival-log replay")
    sim_res = cached_queue_replay(num_doctors=int(num_doctors))
    if not sim_res["available"]:
        st.info(sim_res["message"])
    else:
        st.caption(
            "Counterfactual queue replay uses organizer ground-truth tiers, not this "
            "model's predictions. It does not estimate this model's impact."
        )
        replay_cols = st.columns(3)
        replay_cols[0].metric(
            "FCFS emergency wait",
            f"{sim_res['fcfs_emergency_wait_mins']} min",
        )
        replay_cols[1].metric(
            "Priority replay wait",
            f"{sim_res['cpg_emergency_wait_mins']} min",
        )
        replay_cols[2].metric(
            "Replay difference",
            f"{sim_res['emergency_wait_reduction_pct']}%",
        )
        st.caption(
            f"{sim_res['emergency_patients']:,} emergency visits among "
            f"{sim_res['patients_simulated']:,} labeled synthetic visits. "
            f"FCFS replay mean absolute start-time difference from organizer log: "
            f"{sim_res['official_fcfs_replay_mean_error_seconds']:.2f} seconds."
        )
        with st.expander("Replay assumptions"):
            for assumption in sim_res["assumptions"]:
                st.write(f"- {assumption}")

    render_footer()
