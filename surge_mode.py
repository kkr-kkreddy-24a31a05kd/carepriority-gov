"""CarePriority Gov - Hospital Surge Mode Management Module
=========================================================
Monitors OPD capacity against configurable clinical surge thresholds,
triggers high-visibility command alerts, and maintains operational
surge action checklists for charge nurses and hospital leadership.
"""

from typing import List, Dict, Any, Tuple


DEFAULT_SURGE_CONFIG = {
    "waiting_patient_threshold": 15,
    "emergent_patient_threshold": 3,
    "max_wait_minutes_threshold": 120
}

SURGE_CHECKLIST_TEMPLATE = [
    {
        "id": "cmo_alert",
        "label": "🚨 Mobilize Reserve Physician: Notify on-duty CMO to activate backup doctor desk.",
        "priority": "Critical"
    },
    {
        "id": "fast_track",
        "label": "⚡ Open Fast-Track Counter: Divert Level 4 & 5 non-urgent walk-ins to expedited desk.",
        "priority": "High"
    },
    {
        "id": "rapid_revitals",
        "label": "🩺 Rapid Re-Vitals Sweep: Check SpO2 & BP for all Level 2/3 patients waiting > 15 mins.",
        "priority": "Critical"
    },
    {
        "id": "oxygen_check",
        "label": "💨 Resuscitation Readiness: Confirm central oxygen pressure and crash cart defibrillator.",
        "priority": "Critical"
    },
    {
        "id": "pharmacy_divert",
        "label": "💊 Pharmacy Diversion: Route routine repeat prescription walk-ins to OPD Pharmacy Annex.",
        "priority": "Standard"
    },
    {
        "id": "hallway_marshals",
        "label": "🛡️ Crowd De-escalation: Deploy triage marshals to manage waiting hall spacing.",
        "priority": "Standard"
    }
]


def evaluate_surge_status(
    encounters: List[Dict[str, Any]],
    waiting_threshold: int = 15,
    emergent_threshold: int = 3,
    force_surge: bool = False
) -> Dict[str, Any]:
    """Evaluates whether current OPD workload triggers hospital surge protocols."""
    waiting_count = 0
    emergent_count = 0
    reasons = []

    for enc in encounters:
        st_val = enc.get("status", "Waiting")
        if st_val in ["Waiting", "Awaiting Nurse Review"]:
            waiting_count += 1
            tier = enc.get("tier")
            if tier in [1, 2]:
                emergent_count += 1

    if force_surge:
        reasons.append("Manual Surge Override engaged by Nurse In-Charge / CMO.")

    if waiting_count >= waiting_threshold:
        reasons.append(
            f"Active OPD queue volume ({waiting_count}) reached or exceeded surge threshold ({waiting_threshold})."
        )

    if emergent_count >= emergent_threshold:
        reasons.append(
            f"High-acuity waiting patients ({emergent_count} in L1/L2) exceeded emergent safety limit ({emergent_threshold})."
        )

    is_active = force_surge or (waiting_count >= waiting_threshold) or (emergent_count >= emergent_threshold)

    severity_label = "NORMAL"
    if is_active:
        if emergent_count >= emergent_threshold * 2 or waiting_count >= waiting_threshold * 1.5:
            severity_label = "LEVEL 2 RED SURGE"
        else:
            severity_label = "LEVEL 1 AMBER SURGE"

    return {
        "is_active": is_active,
        "severity": severity_label,
        "waiting_count": waiting_count,
        "emergent_count": emergent_count,
        "waiting_threshold": waiting_threshold,
        "emergent_threshold": emergent_threshold,
        "reasons": reasons
    }


def render_surge_banner_html(surge_info: Dict[str, Any]) -> str:
    """Generates accessible, high-visibility 3D surge banner markup."""
    if not surge_info.get("is_active"):
        return ""

    severity = surge_info.get("severity", "LEVEL 1 AMBER SURGE")
    is_red = "RED" in severity
    bg_gradient = "linear-gradient(135deg, rgba(255, 42, 85, 0.28) 0%, rgba(30, 8, 16, 0.95) 100%)" if is_red else "linear-gradient(135deg, rgba(255, 122, 0, 0.28) 0%, rgba(35, 18, 5, 0.95) 100%)"
    border_color = "#FF2A55" if is_red else "#FF7A00"
    glow_color = "rgba(255, 42, 85, 0.4)" if is_red else "rgba(255, 122, 0, 0.4)"
    title_text = "🚨 CRITICAL HOSPITAL OVERLOAD: " + severity if is_red else "⚠️ OPD SURGE PROTOCOL ACTIVATED: " + severity

    reasons_li = "".join([f"<li>{r}</li>" for r in surge_info.get("reasons", [])])

    return f"""
    <div class="surge-mode-banner" style="
      background: {bg_gradient};
      border: 2px solid {border_color};
      border-left: 8px solid {border_color};
      border-radius: 18px;
      padding: 18px 24px;
      margin-bottom: 22px;
      box-shadow: 0 10px 32px {glow_color}, inset 0 1px 0 rgba(255, 255, 255, 0.2);
      animation: surgeGlowPulse 2.4s infinite alternate;
      position: relative;
      overflow: hidden;
    ">
      <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; margin-bottom: 10px;">
        <div style="font-family: 'Sora', sans-serif; font-size: 1.15rem; font-weight: 800; color: #FFFFFF; letter-spacing: 0.02em;">
          {title_text}
        </div>
        <div style="display: flex; gap: 10px;">
          <span style="background: rgba(0,0,0,0.4); border: 1px solid {border_color}; color: #FFFFFF; font-size: 0.8rem; font-weight: 700; padding: 4px 12px; border-radius: 12px;">
            Waiting: {surge_info.get('waiting_count', 0)} / Limit {surge_info.get('waiting_threshold', 15)}
          </span>
          <span style="background: rgba(0,0,0,0.4); border: 1px solid {border_color}; color: #FFFFFF; font-size: 0.8rem; font-weight: 700; padding: 4px 12px; border-radius: 12px;">
            L1/L2 Emergent: {surge_info.get('emergent_count', 0)} / Limit {surge_info.get('emergent_threshold', 3)}
          </span>
        </div>
      </div>
      <div style="font-size: 0.88rem; color: #F1F5F9; line-height: 1.45;">
        <ul style="margin: 0 0 8px 18px; padding: 0;">
          {reasons_li}
        </ul>
        <span style="font-size: 0.8rem; color: #FFD13B; font-weight: 600;">
          📋 Action Required: Execute the Surge Protocol Checklist below. Mobilize clinical reserves immediately.
        </span>
      </div>
    </div>
    """
