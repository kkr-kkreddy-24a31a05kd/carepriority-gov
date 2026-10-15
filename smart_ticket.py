"""CarePriority Gov - Smart Token & QR Ticket Generation Module
=============================================================
Generates official, digital/printable OPD Smart Tokens with embedded
dynamic QR codes, queue position, estimated wait times, and bilingual
emergency escalation disclaimers.

PRIVACY BY DEFAULT:
Patient-facing tokens omit priority tier, patient name, and vital signs
to prevent waiting room anxiety, queue disputes, and HIPAA/DISHA data leaks.
Staff copies can optionally include clinical details with explicit authorization.
"""

import io
import json
import base64
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional

try:
    import qrcode
    HAS_QRCODE = True
except ImportError:
    HAS_QRCODE = False

from styles import render_tier_badge


def generate_qr_base64(data_payload: Dict[str, Any]) -> str:
    """Generates a high-contrast QR code as a base64 PNG data-URI."""
    json_str = json.dumps(data_payload, separators=(',', ':'))
    
    if HAS_QRCODE:
        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=5,
            border=2,
        )
        qr.add_data(json_str)
        qr.make(fit=True)
        img = qr.make_image(fill_color="#050E1F", back_color="#FFFFFF")
        
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    else:
        # Fallback placeholder if qrcode is unavailable
        svg = """<svg width="140" height="140" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
          <rect width="100" height="100" fill="#FFFFFF"/>
          <rect x="10" y="10" width="30" height="30" fill="#050E1F"/>
          <rect x="60" y="10" width="30" height="30" fill="#050E1F"/>
          <rect x="10" y="60" width="30" height="30" fill="#050E1F"/>
          <rect x="45" y="45" width="10" height="10" fill="#087F8C"/>
          <text x="50" y="55" font-size="8" fill="#087F8C" text-anchor="middle">QR CODE</text>
        </svg>"""
        b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
        return f"data:image/svg+xml;base64,{b64}"


def render_smart_token_ticket(
    encounter: Dict[str, Any],
    queue_pos: int = 1,
    est_wait_mins: int = 15,
    show_staff_clinical_info: bool = False
) -> str:
    """
    Renders a healthcare command center digital Smart Token Ticket HTML
    complete with QR code, queue telemetry, and safety notices.

    By default (`show_staff_clinical_info=False`), patient name, priority tier,
    and vital signs are OMITTED from both the printed card and QR payload to protect
    patient privacy and prevent triage-tier anxiety.
    """
    enc_id = encounter.get("encounter_id", "ENC-0000")
    # Generate short token number e.g. T-0042
    short_token = enc_id.split("-")[-1] if "-" in enc_id else enc_id[-4:]
    token_display = f"T-{short_token}"

    patient_name = encounter.get("patient_name") or f"Walk-in {enc_id}"
    age = encounter.get("age")
    age_str = f"{age}y" if age is not None else "Age Unknown"
    gender = encounter.get("gender", "Unknown")
    dept = encounter.get("department", "General Medicine")
    tier = encounter.get("tier")
    abstained = encounter.get("abstained", False)
    created_at = encounter.get("created_at", datetime.now().isoformat())

    try:
        dt = datetime.fromisoformat(created_at)
        time_str = dt.strftime("%I:%M %p &bull; %d %b %Y")
    except Exception:
        time_str = created_at

    # Build QR payload
    security_hash = hashlib.sha256(f"{enc_id}:{created_at}".encode()).hexdigest()[:10]
    
    if show_staff_clinical_info:
        # Internal staff copy: includes clinical triage details
        qr_payload = {
            "cpg_token": token_display,
            "encounter_id": enc_id,
            "patient": patient_name,
            "tier": f"L{tier}" if tier else "NEEDS_REVIEW",
            "dept": dept,
            "auth": security_hash,
            "issued": created_at
        }
    else:
        # Patient-facing token: NO tier, NO name, NO vitals
        qr_payload = {
            "cpg_token": token_display,
            "encounter_id": enc_id,
            "dept": dept,
            "pos": queue_pos,
            "auth": security_hash,
            "issued": created_at
        }
        
    qr_data_uri = generate_qr_base64(qr_payload)
    tier_badge_html = render_tier_badge(tier, abstained)

    v = encounter.get("vitals", {})
    vitals_disp = (
        f"SpO2: {v.get('spo2') or 'N/A'}% | "
        f"BP: {v.get('bp_systolic') or 'N/A'}/{v.get('bp_diastolic') or 'N/A'} | "
        f"Pulse: {v.get('pulse') or 'N/A'} bpm | "
        f"Temp: {v.get('temperature') or 'N/A'}°F"
    )

    if show_staff_clinical_info:
        header_subtitle = "Clinical Chart Copy &bull; Internal Staff Authorization"
        token_badge_html = tier_badge_html
        patient_section = f"""
        <div style="font-size: 0.95rem; color: #FFFFFF; font-weight: 700; margin-bottom: 4px;">
          {patient_name} <span style="font-size: 0.84rem; color: #94A3B8; font-weight: 500;">({age_str} &bull; {gender})</span>
        </div>
        """
        vitals_section = f"""
        <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 6px;">
          <strong>Intake Vitals:</strong> {vitals_disp}
        </div>
        """
        qr_label = "Doctor Desk Scan (Staff)"
    else:
        header_subtitle = "Patient OPD Smart Pass &bull; General Registration"
        token_badge_html = """<span style="background: rgba(8, 127, 140, 0.25); border: 1px solid #66E3C4; color: #66E3C4; font-family: 'Sora', sans-serif; font-weight: 700; font-size: 0.82rem; padding: 5px 12px; border-radius: 20px; display: inline-flex; align-items: center; gap: 6px;">
          🏥 OPD Registered &bull; Active Queue
        </span>"""
        patient_section = f"""
        <div style="font-size: 0.95rem; color: #66E3C4; font-weight: 700; margin-bottom: 4px;">
          Pass Identifier: <span style="color: #FFFFFF;">#{token_display}</span>
        </div>
        """
        vitals_section = ""  # Strictly keep vitals off patient-facing token
        qr_label = "Scan at Doctor Desk"

    ticket_html = f"""
    <div class="smart-token-ticket" style="
      background: linear-gradient(135deg, rgba(8, 22, 42, 0.95) 0%, rgba(5, 14, 31, 0.98) 100%);
      border: 2px dashed rgba(102, 227, 196, 0.45);
      border-radius: 18px;
      padding: 24px;
      margin: 16px 0 24px 0;
      box-shadow: 0 16px 36px rgba(0, 0, 0, 0.6), inset 0 1px 1px rgba(255, 255, 255, 0.15);
      position: relative;
      overflow: hidden;
    ">
      <!-- Watermark Background -->
      <div style="position: absolute; right: -20px; bottom: -20px; font-size: 8rem; opacity: 0.03; font-weight: 900; pointer-events: none; font-family: 'Sora', sans-serif;">
        OPD
      </div>

      <!-- Top Header Row -->
      <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255, 255, 255, 0.1); padding-bottom: 14px; margin-bottom: 16px; flex-wrap: wrap; gap: 10px;">
        <div style="display: flex; align-items: center; gap: 10px;">
          <div style="font-size: 1.6rem;">🏥</div>
          <div>
            <div style="font-family: 'Sora', sans-serif; font-size: 0.78rem; font-weight: 800; color: #66E3C4; text-transform: uppercase; letter-spacing: 0.1em;">
              Government District Hospital &bull; OPD Command Triage
            </div>
            <div style="font-family: 'Sora', sans-serif; font-size: 1.25rem; font-weight: 800; color: #FFFFFF;">
              CarePriority Digital OPD Smart Pass
            </div>
            <div style="font-size: 0.74rem; color: #94A3B8; margin-top: 2px;">
              {header_subtitle}
            </div>
          </div>
        </div>
        <div style="text-align: right;">
          <div style="font-size: 0.76rem; color: #94A3B8;">Issued Timestamp:</div>
          <div style="font-size: 0.84rem; color: #CBD5E1; font-weight: 600;">{time_str}</div>
        </div>
      </div>

      <!-- Main Ticket Body: Token + QR + Clinical Details -->
      <div style="display: grid; grid-template-columns: 1.8fr 1fr; gap: 20px; align-items: center; margin-bottom: 18px;">
        <div>
          <!-- Token Number Spotlight -->
          <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px; flex-wrap: wrap;">
            <div style="
              background: linear-gradient(135deg, #087F8C, #0E4A6F);
              color: #FFFFFF;
              font-family: 'Sora', sans-serif;
              font-size: 1.9rem;
              font-weight: 800;
              padding: 6px 18px;
              border-radius: 14px;
              border: 1px solid #66E3C4;
              letter-spacing: 0.05em;
              box-shadow: 0 4px 14px rgba(8, 127, 140, 0.4);
            ">
              TOKEN #{token_display}
            </div>
            {token_badge_html}
          </div>

          <!-- Patient & Routing Info -->
          {patient_section}
          <div style="font-size: 0.84rem; color: #CBD5E1; margin-bottom: 10px;">
            Encounter ID: <code>{enc_id}</code> &bull; Verification Hash: <code>{security_hash}</code>
          </div>

          <!-- Queue Telemetry Strip -->
          <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 12px;">
            <span style="background: rgba(10, 26, 47, 0.8); border: 1px solid rgba(102, 227, 196, 0.4); color: #66E3C4; font-size: 0.82rem; font-weight: 700; padding: 4px 12px; border-radius: 12px;">
              🏢 Routing: {dept}
            </span>
            <span style="background: rgba(10, 26, 47, 0.8); border: 1px solid rgba(56, 189, 248, 0.4); color: #38BDF8; font-size: 0.82rem; font-weight: 700; padding: 4px 12px; border-radius: 12px;">
              📍 Position in Queue: #{queue_pos}
            </span>
            <span style="background: rgba(10, 26, 47, 0.8); border: 1px solid rgba(255, 209, 59, 0.4); color: #FFD13B; font-size: 0.82rem; font-weight: 700; padding: 4px 12px; border-radius: 12px;">
              ⏱️ Est. Wait: ~{est_wait_mins}m
            </span>
          </div>

          {vitals_section}
        </div>

        <!-- High-Contrast QR Code Card -->
        <div style="
          background: #FFFFFF;
          padding: 12px;
          border-radius: 14px;
          box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
          text-align: center;
          max-width: 175px;
          margin: 0 auto;
        ">
          <img src="{qr_data_uri}" alt="Smart Token QR Code" style="width: 140px; height: 140px; display: block; margin: 0 auto;" />
          <div style="color: #050E1F; font-size: 0.68rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 6px;">
            {qr_label}
          </div>
        </div>
      </div>

      <!-- Bilingual Clinical Safety Notice -->
      <div style="
        background: rgba(255, 42, 85, 0.12);
        border: 1px solid rgba(255, 42, 85, 0.4);
        border-radius: 12px;
        padding: 10px 16px;
        font-size: 0.82rem;
        color: #FFE6EB;
        line-height: 1.45;
      ">
        <strong>⚠️ EMERGENCY WARNING / आपातकालीन सूचना:</strong><br>
        If your condition worsens while waiting (sudden chest pain, breathing difficulty, extreme dizziness, or fainting), <strong>notify the triage nurse desk immediately!</strong><br>
        <span style="font-size: 0.78rem; opacity: 0.9;">यदि प्रतीक्षा करते समय आपकी स्थिति बिगड़ती है (सीने में दर्द, सांस लेने में तकलीफ), तो तुरंत ट्राइएज नर्स को सूचित करें।</span>
      </div>
    </div>
    """
    return ticket_html
