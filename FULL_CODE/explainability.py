"""CarePriority Gov - Explainability Visuals Module
=================================================
Generates factual, transparent clinical explainability views for triage assessments.

KEY CONSTRAINTS & PRINCIPLES:
1. Strictly Real Computed Reasons Only:
   Every row represents a real contributing factor, displaying:
   [Actual Measured Value] -> [Matched Rule/Threshold from rules_config.py] -> [Clinical Acuity Effect]
   (e.g., "SpO2 = 88% -> below 92.0% -> escalated to at least L2").
2. Strict Ordering by Effect:
   1. Hard escalation rules first (mandatory physiological triggers from HARD_ESCALATION_VITALS)
   2. Red-flag keywords next (from RED_FLAG_KEYWORDS)
   3. Out-of-range vitals next (borderline/abnormal physiological parameters)
   4. Model output last (predicted tier and real predict_proba probabilities)
3. Zero Invented Numbers / Confidence Scores:
   Arbitrary hard-coded severity weights (e.g. 85%, 90%) are strictly prohibited.
   Percentage bars are ONLY rendered for the model output, where each value comes
   directly from the mathematical predict_proba array.
4. Active Dataset Labeling:
   Labeled "model output (trained on placeholder data)" while placeholder data is active.
"""

from typing import Dict, List, Any, Optional
import numpy as np

from rules_config import (
    TRIAGE_LEVELS,
    RED_FLAG_KEYWORDS,
    SECONDARY_KEYWORDS
)
import evaluate_runner
from triage_engine import (
    detect_stage1_canonical_red_flags,
    detect_stage1_red_flag_keyword_matches,
    detect_symptom_matches,
    evaluate_hard_escalations,
    predict_ml_triage,
)
from styles import clean_html


def compute_factual_contributions(encounter: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts real contributing factors ordered strictly by effect:
    1. Hard escalation rules first
    2. Red-flag keywords next
    3. Out-of-range vitals next
    4. Model output last (real predict_proba array)
    """
    v = encounter.get("vitals") or {}
    spo2 = v.get("spo2")
    sbp = v.get("bp_systolic")
    dbp = v.get("bp_diastolic")
    pulse = v.get("pulse")
    temp = v.get("temperature")
    complaint = (encounter.get("complaint_original") or encounter.get("complaint") or "")
    complaint_lower = complaint.lower()

    # -------------------------------------------------------------------------
    # 1. HARD ESCALATION RULES FIRST (Mandatory Physiological Triggers)
    # -------------------------------------------------------------------------
    hard_escalations: List[Dict[str, str]] = []
    hard_tier, _, hard_reasons, detected_flags = evaluate_hard_escalations(
        spo2=spo2,
        bp_systolic=sbp,
        bp_diastolic=dbp,
        pulse=pulse,
        temperature=temp,
        complaint=complaint,
        age=encounter.get("age"),
        age_months=encounter.get("age_months"),
        pregnant=encounter.get("pregnant"),
        pregnancy_stage=encounter.get("pregnancy_stage"),
        recently_delivered=bool(encounter.get("recently_delivered")),
        known_conditions=encounter.get("known_conditions"),
        medications=encounter.get("medications")
    )
    hard_badge = "#FF2A55" if hard_tier == 1 else "#FF7A00"
    for reason in hard_reasons:
        hard_escalations.append({
            "factor": "Primer-aligned hard rule",
            "measured": reason,
            "matched_rule": "evaluated by the active triage_engine rules",
            "effect": f"escalated to L{hard_tier}",
            "badge_color": hard_badge,
            "statement": reason
        })

    # -------------------------------------------------------------------------
    # 2. RED-FLAG KEYWORDS NEXT (Complaint Term Matches)
    # -------------------------------------------------------------------------
    red_flag_factors: List[Dict[str, str]] = []
    l1_matches = detect_stage1_red_flag_keyword_matches(complaint, "L1")
    for kw in l1_matches:
        red_flag_factors.append({
            "factor": "Chief Complaint Keyword (L1)",
            "measured": f'Keyword = "{kw}"',
            "matched_rule": 'matched RED_FLAG_KEYWORDS["L1_SYMPTOMS"]',
            "effect": "escalated to L1 (Resuscitation)",
            "badge_color": "#FF2A55",
            "statement": f'Keyword = "{kw}" → matched RED_FLAG_KEYWORDS["L1_SYMPTOMS"] → escalated to L1 (Resuscitation)'
        })

    l2_matches = detect_stage1_red_flag_keyword_matches(complaint, "L2")
    for kw in l2_matches:
        red_flag_factors.append({
            "factor": "Chief Complaint Keyword (L2)",
            "measured": f'Keyword = "{kw}"',
            "matched_rule": 'matched RED_FLAG_KEYWORDS["L2_SYMPTOMS"]',
            "effect": "escalated to at least L2 (Emergent)",
            "badge_color": "#FF7A00",
            "statement": f'Keyword = "{kw}" → matched RED_FLAG_KEYWORDS["L2_SYMPTOMS"] → escalated to at least L2 (Emergent)'
        })
    _, canonical_level1_flags, _ = detect_stage1_canonical_red_flags(complaint)
    for flag in detected_flags:
        if flag.startswith("RF_"):
            effect = (
                "escalated to L1 (Resuscitation)"
                if flag in canonical_level1_flags
                else "escalated to at least L2 (Emergent)"
            )
            red_flag_factors.append({
                "factor": "Contextual red-flag code",
                "measured": flag,
                "matched_rule": "primer symptom/context rule",
                "effect": effect,
                "badge_color": "#FF2A55" if flag in canonical_level1_flags else "#FF7A00",
                "statement": f"{flag} → contextual primer rule → {effect}"
            })

    # -------------------------------------------------------------------------
    # 3. OUT-OF-RANGE VITALS NEXT (Borderline/Abnormal Non-Hard Escalations)
    # -------------------------------------------------------------------------
    out_of_range: List[Dict[str, str]] = []

    # Primer amber SpO2.
    if spo2 is not None and 92.0 <= spo2 <= 93.0:
        out_of_range.append({
            "factor": "Oxygen Saturation (SpO2)",
            "measured": f"SpO2 = {spo2}%",
            "matched_rule": "92%–93% (primer amber range)",
            "effect": "contributes to L3 (Urgent)",
            "badge_color": "#FFC700",
            "statement": f"SpO2 = {spo2}% → 92%–93% → amber vital (contributes to L3 Urgent)"
        })

    # Pregnancy-specific amber blood pressure.
    if encounter.get("pregnant") is True and sbp is not None and 140 <= sbp < 160:
        out_of_range.append({
            "factor": "Pregnancy systolic blood pressure",
            "measured": f"SBP = {sbp} mmHg",
            "matched_rule": "140–159 mmHg (primer amber range)",
            "effect": "contributes to L3 (Urgent)",
            "badge_color": "#FFC700",
            "statement": f"Pregnant patient SBP = {sbp} mmHg → 140–159 mmHg → amber vital"
        })
    if encounter.get("pregnant") is True and dbp is not None and 90 <= dbp < 110:
        out_of_range.append({
            "factor": "Pregnancy diastolic blood pressure",
            "measured": f"DBP = {dbp} mmHg",
            "matched_rule": "90–109 mmHg (primer amber range)",
            "effect": "contributes to L3 (Urgent)",
            "badge_color": "#FFC700",
            "statement": f"Pregnant patient DBP = {dbp} mmHg → 90–109 mmHg → amber vital"
        })

    # Adult amber pulse.
    age_years = (
        encounter.get("age_months") / 12
        if encounter.get("age_months") is not None
        else encounter.get("age")
    )
    if age_years is not None and age_years >= 16 and pulse is not None:
        if 111 <= pulse <= 130:
            out_of_range.append({
                "factor": "Pulse Rate",
                "measured": f"Pulse = {pulse} bpm",
                "matched_rule": "111–130 bpm (adult primer amber range)",
                "effect": "contributes to L3 (Urgent)",
                "badge_color": "#FFC700",
                "statement": f"Pulse = {pulse} bpm → 111–130 bpm → adult amber vital"
            })

    # Primer amber temperature for patients aged 3 months or older.
    age_months = encounter.get("age_months")
    if age_months is None and age_years is not None:
        age_months = age_years * 12
    temp_c = (temp - 32) * 5 / 9 if temp is not None else None
    if temp_c is not None and age_months is not None and age_months >= 3 and temp_c >= 39.1:
        out_of_range.append({
            "factor": "Body Temperature",
            "measured": f"Temp = {temp_c:.1f}°C",
            "matched_rule": "≥ 39.1°C at age 3 months or older",
            "effect": "contributes to L3 (Urgent)",
            "badge_color": "#FFC700",
            "statement": f"Temp = {temp_c:.1f}°C → ≥ 39.1°C → amber vital"
        })

    # Secondary symptom keywords
    l3_secondary = detect_symptom_matches(complaint_lower, SECONDARY_KEYWORDS["L3_URGENT"])
    for kw in l3_secondary:
        out_of_range.append({
            "factor": "Secondary Symptom Keyword",
            "measured": f'Keyword = "{kw}"',
            "matched_rule": 'matched SECONDARY_KEYWORDS["L3_URGENT"]',
            "effect": "contributes to L3 (Urgent) differentiation",
            "badge_color": "#FFC700",
            "statement": f'Keyword = "{kw}" → matched SECONDARY_KEYWORDS["L3_URGENT"] → contributes to L3 (Urgent)'
        })

    l4_secondary = detect_symptom_matches(complaint_lower, SECONDARY_KEYWORDS["L4_LESS_URGENT"])
    for kw in l4_secondary:
        out_of_range.append({
            "factor": "Secondary Symptom Keyword",
            "measured": f'Keyword = "{kw}"',
            "matched_rule": 'matched SECONDARY_KEYWORDS["L4_LESS_URGENT"]',
            "effect": "contributes to L4 (Less Urgent)",
            "badge_color": "#00A3FF",
            "statement": f'Keyword = "{kw}" → matched SECONDARY_KEYWORDS["L4_LESS_URGENT"] → contributes to L4 (Less Urgent)'
        })

    l5_secondary = detect_symptom_matches(complaint_lower, SECONDARY_KEYWORDS["L5_NON_URGENT"])
    for kw in l5_secondary:
        out_of_range.append({
            "factor": "Secondary Symptom Keyword",
            "measured": f'Keyword = "{kw}"',
            "matched_rule": 'matched SECONDARY_KEYWORDS["L5_NON_URGENT"]',
            "effect": "contributes to L5 (Non-Urgent)",
            "badge_color": "#00D68F",
            "statement": f'Keyword = "{kw}" → matched SECONDARY_KEYWORDS["L5_NON_URGENT"] → contributes to L5 (Non-Urgent)'
        })

    # -------------------------------------------------------------------------
    # 4. MODEL OUTPUT LAST (Predicted Tier & Real predict_proba Distribution)
    # -------------------------------------------------------------------------
    is_placeholder = encounter.get("is_placeholder", True)
    if encounter.get("is_organizer_data") is True:
        is_placeholder = False
    model_label = "model output" if not is_placeholder else "model output (trained on placeholder data)"

    model_tier = encounter.get("model_tier")
    raw_probs = encounter.get("model_probs")

    # If model_probs was not pre-stored on the encounter document, compute live
    if raw_probs is None and complaint:
        try:
            p_tier, p_conf, p_probs, _ = predict_ml_triage(
                complaint=complaint,
                age=encounter.get("age"),
                gender=encounter.get("gender"),
                spo2=spo2,
                bp_systolic=sbp,
                bp_diastolic=dbp,
                pulse=pulse,
                temperature=temp,
                known_conditions=encounter.get("known_conditions"),
                medications=encounter.get("medications")
            )
            if p_probs is not None:
                raw_probs = [float(x) for x in p_probs]
            if model_tier is None:
                model_tier = p_tier
        except Exception:
            raw_probs = None

    model_data = {
        "label": model_label,
        "predicted_tier": model_tier,
        "probabilities": raw_probs,  # Real array from predict_proba (no invented numbers)
        "confidence": encounter.get("model_confidence")
    }

    return {
        "hard_escalations": hard_escalations,
        "red_flag_factors": red_flag_factors,
        "out_of_range_vitals": out_of_range,
        "model_output": model_data
    }


def render_horizontal_contribution_view(encounter: Dict[str, Any]) -> str:
    """
    Renders a purely factual, transparent explainability card for triage assessments.
    Strictly eliminates invented percentages. Only real mathematical probabilities
    from predict_proba are displayed as percentage bars.
    """
    contributions = compute_factual_contributions(encounter)
    hard_escalations = contributions["hard_escalations"]
    red_flag_factors = contributions["red_flag_factors"]
    out_of_range = contributions["out_of_range_vitals"]
    model_output = contributions["model_output"]

    # --- Section 1: Hard Escalation Rules ---
    if hard_escalations:
        hard_rows = "".join([
            f'<div style="background: rgba(255, 42, 85, 0.08); border-left: 3px solid {item["badge_color"]}; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">'
            f'<div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 8px;">'
            f'<span style="font-weight: 700; color: #FFFFFF; font-size: 0.88rem;">{item["factor"]}</span>'
            f'<span style="font-size: 0.76rem; font-weight: 700; color: {item["badge_color"]}; background: rgba(255,42,85,0.15); padding: 2px 10px; border-radius: 12px; border: 1px solid {item["badge_color"]}44;">{item["effect"]}</span>'
            f'</div>'
            f'<div style="font-family: \'JetBrains Mono\', monospace; font-size: 0.82rem; color: #CBD5E1; margin-top: 5px;">'
            f'<span style="color: #66E3C4;">{item["measured"]}</span> <span style="color: #94A3B8;">&rarr;</span> <span style="color: #FFD13B;">{item["matched_rule"]}</span> <span style="color: #94A3B8;">&rarr;</span> <span style="color: {item["badge_color"]};">{item["effect"]}</span>'
            f'</div>'
            f'</div>'
            for item in hard_escalations
        ])
    else:
        hard_rows = '<div style="font-size: 0.82rem; color: #94A3B8; font-style: italic; padding: 4px 0;">✓ None triggered (all vital signs within non-critical parameters)</div>'

    # --- Section 2: Red-Flag Keywords ---
    if red_flag_factors:
        rf_rows = "".join([
            f'<div style="background: rgba(255, 122, 0, 0.08); border-left: 3px solid {item["badge_color"]}; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">'
            f'<div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 8px;">'
            f'<span style="font-weight: 700; color: #FFFFFF; font-size: 0.88rem;">{item["factor"]}</span>'
            f'<span style="font-size: 0.76rem; font-weight: 700; color: {item["badge_color"]}; background: rgba(255,122,0,0.15); padding: 2px 10px; border-radius: 12px; border: 1px solid {item["badge_color"]}44;">{item["effect"]}</span>'
            f'</div>'
            f'<div style="font-family: \'JetBrains Mono\', monospace; font-size: 0.82rem; color: #CBD5E1; margin-top: 5px;">'
            f'<span style="color: #66E3C4;">{item["measured"]}</span> <span style="color: #94A3B8;">&rarr;</span> <span style="color: #FFD13B;">{item["matched_rule"]}</span> <span style="color: #94A3B8;">&rarr;</span> <span style="color: {item["badge_color"]};">{item["effect"]}</span>'
            f'</div>'
            f'</div>'
            for item in red_flag_factors
        ])
    else:
        rf_rows = '<div style="font-size: 0.82rem; color: #94A3B8; font-style: italic; padding: 4px 0;">✓ None detected (no emergent red-flag keywords in complaint)</div>'

    # --- Section 3: Out-of-Range Vitals ---
    if out_of_range:
        oor_rows = "".join([
            f'<div style="background: rgba(255, 199, 0, 0.06); border-left: 3px solid {item["badge_color"]}; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">'
            f'<div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 8px;">'
            f'<span style="font-weight: 700; color: #FFFFFF; font-size: 0.88rem;">{item["factor"]}</span>'
            f'<span style="font-size: 0.76rem; font-weight: 700; color: {item["badge_color"]}; background: rgba(255,199,0,0.15); padding: 2px 10px; border-radius: 12px; border: 1px solid {item["badge_color"]}44;">{item["effect"]}</span>'
            f'</div>'
            f'<div style="font-family: \'JetBrains Mono\', monospace; font-size: 0.82rem; color: #CBD5E1; margin-top: 5px;">'
            f'<span style="color: #66E3C4;">{item["measured"]}</span> <span style="color: #94A3B8;">&rarr;</span> <span style="color: #FFD13B;">{item["matched_rule"]}</span> <span style="color: #94A3B8;">&rarr;</span> <span style="color: {item["badge_color"]};">{item["effect"]}</span>'
            f'</div>'
            f'</div>'
            for item in out_of_range
        ])
    else:
        oor_rows = '<div style="font-size: 0.82rem; color: #94A3B8; font-style: italic; padding: 4px 0;">✓ None (all recorded vital signs within normal physiological ranges)</div>'

    # --- Section 4: Model Output (Real predict_proba Probabilities) ---
    pred_tier = model_output["predicted_tier"]
    probs = model_output["probabilities"]
    model_label = model_output["label"]

    tier_prob_rows = []
    if probs is not None and len(probs) >= 5:
        tier_colors = ["#FF2A55", "#FF7A00", "#FFC700", "#00A3FF", "#00D68F"]
        for t_idx in range(5):
            t_num = t_idx + 1
            p_val = float(probs[t_idx])
            p_pct = round(p_val * 100.0, 1)
            t_color = tier_colors[t_idx]
            t_name = TRIAGE_LEVELS[t_num]["short_name"]
            is_pred = (pred_tier == t_num)
            marker = " (ML Proposed)" if is_pred else ""

            row_html = (
                f'<div style="margin-bottom: 8px;">'
                f'<div style="display: flex; justify-content: space-between; font-size: 0.78rem; color: #E2E8F0; margin-bottom: 3px;">'
                f'<span><strong>{t_name}</strong>{marker}</span>'
                f'<span style="font-family: \'JetBrains Mono\', monospace; font-weight: 700; color: {t_color};">{p_pct:.1f}%</span>'
                f'</div>'
                f'<div style="width: 100%; height: 6px; background: rgba(255,255,255,0.06); border-radius: 3px; overflow: hidden;">'
                f'<div style="width: {p_pct}%; height: 100%; background: {t_color}; border-radius: 3px; transition: width 0.3s ease;"></div>'
                f'</div>'
                f'</div>'
            )
            tier_prob_rows.append(row_html)
        model_distribution_html = "".join(tier_prob_rows)
    else:
        pred_label = f"Level {pred_tier}" if pred_tier else "Not Available"
        model_distribution_html = f'<div style="font-size: 0.82rem; color: #94A3B8; font-style: italic;">Predicted Tier: {pred_label} (model probability vector not loaded)</div>'

    markup = f"""
    <div class="glass-card" style="margin-top: 14px; border-left: 4px solid #38BDF8;">
      <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 14px;">
        <div style="font-family: 'Sora', sans-serif; font-size: 1.05rem; font-weight: 800; color: #FFFFFF;">
          ⚖️ Acuity Factor Contribution Breakdown (Why this priority?)
        </div>
        <div style="font-size: 0.72rem; color: #66E3C4; background: rgba(0,214,143,0.12); border: 1px solid rgba(0,214,143,0.3); padding: 3px 10px; border-radius: 16px;">
          ✓ Factual View &bull; Zero Invented Percentages
        </div>
      </div>

      <!-- 1. Hard Escalation Rules -->
      <div style="margin-bottom: 14px;">
        <div class="label-caps" style="color: #FF8099; margin-bottom: 6px;">1. Hard Escalation Rules (Mandatory Safety Triggers)</div>
        {hard_rows}
      </div>

      <!-- 2. Red-Flag Keywords -->
      <div style="margin-bottom: 14px;">
        <div class="label-caps" style="color: #FFA336; margin-bottom: 6px;">2. Red-Flag Keywords (Complaint Dictionary Matches)</div>
        {rf_rows}
      </div>

      <!-- 3. Out-of-Range Vitals -->
      <div style="margin-bottom: 14px;">
        <div class="label-caps" style="color: #FFD13B; margin-bottom: 6px;">3. Out-of-Range Physiological Vitals</div>
        {oor_rows}
      </div>

      <!-- 4. Model Output -->
      <div style="margin-top: 16px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.08);">
        <div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; margin-bottom: 8px;">
          <div class="label-caps" style="color: #38BDF8;">4. {model_label.upper()}</div>
          <div style="font-size: 0.78rem; color: #CBD5E1;">
            Model Proposed Tier: <strong style="color: #66E3C4;">Level {pred_tier if pred_tier else 'N/A'}</strong>
          </div>
        </div>
        <div style="font-size: 0.76rem; color: #94A3B8; margin-bottom: 10px;">
          Real calculated <code>predict_proba</code> distribution across all 5 clinical tiers:
        </div>
        <div>
          {model_distribution_html}
        </div>
      </div>
    </div>
    """
    return clean_html(markup)
