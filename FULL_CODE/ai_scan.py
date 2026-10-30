"""CarePriority Gov - AI Scan & Offline OCR Module
=================================================
Performs offline optical character recognition (OCR) on photographed paper triage
slips, extracts clinical vitals and complaint strings via regex pattern matching,
presents a per-field review table with checkboxes, and applies selected values
to the intake form for validation.

CRITICAL PRIVACY & SECURITY GUARDRAILS:
1. Never store images: images uploaded or captured via camera are processed
   entirely in-memory (RAM) and immediately discarded. No files are saved to disk.
2. No cloud mode: 100% offline local processing only.
3. Safety rule: AI scan only drafts values into the form; the human nurse always
   reviews, edits, and submits through standard validation guardrails.
"""

import io
import re
from typing import Dict, Any, Tuple, Optional, List
from PIL import Image
import streamlit as st
from styles import render_html

try:
    import pytesseract
    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False


def check_ocr_status() -> Tuple[bool, str]:
    """Checks whether an offline OCR engine (pytesseract + tesseract binary) is installed."""
    if not HAS_PYTESSERACT:
        return False, "pytesseract library not installed"
    
    try:
        ver = pytesseract.get_tesseract_version()
        return True, f"Tesseract v{ver} available"
    except Exception:
        return False, "Tesseract OCR binary not found in system PATH"


def parse_clinical_text(raw_text: str) -> Dict[str, Any]:
    """Extracts clinical vital signs and symptoms from OCR transcript using robust regex patterns."""
    result: Dict[str, Any] = {
        "age": None,
        "gender": None,
        "spo2": None,
        "bp_systolic": None,
        "bp_diastolic": None,
        "pulse": None,
        "temperature": None,
        "temp_unit": "F",
        "complaint": "",
        "raw_text": raw_text.strip()
    }

    if not raw_text:
        return result

    # 1. SpO2 Pattern (e.g., SpO2: 94%, O2 Sat 88, SpO2 98)
    spo2_match = re.search(r"(?i)(?:spo2|sp02|o2\s*sat|saturation)\s*[:=]?\s*(\d{2,3})", raw_text)
    if spo2_match:
        try:
            val = float(spo2_match.group(1))
            if 40 <= val <= 100:
                result["spo2"] = val
        except ValueError:
            pass

    # 2. Blood Pressure Pattern (e.g., BP: 140/90, B.P 120/80, BP 130 / 85)
    bp_match = re.search(r"(?i)(?:bp|b\.p\.|blood\s*pressure)\s*[:=]?\s*(\d{2,3})\s*[/xX\-]\s*(\d{2,3})", raw_text)
    if bp_match:
        try:
            sys_val = int(bp_match.group(1))
            dia_val = int(bp_match.group(2))
            if 40 <= sys_val <= 260 and 20 <= dia_val <= 160:
                result["bp_systolic"] = sys_val
                result["bp_diastolic"] = dia_val
        except ValueError:
            pass

    # 3. Pulse / Heart Rate Pattern (e.g., Pulse: 88 bpm, PR 104, HR: 76)
    pulse_match = re.search(r"(?i)(?:pulse|pr|hr|heart\s*rate)\s*[:=]?\s*(\d{2,3})\s*(?:bpm)?", raw_text)
    if pulse_match:
        try:
            val = int(pulse_match.group(1))
            if 30 <= val <= 240:
                result["pulse"] = val
        except ValueError:
            pass

    # 4. Temperature Pattern (e.g., Temp: 98.6 F, Temperature 101.2, Temp 38.5 C)
    temp_match = re.search(r"(?i)(?:temp|temperature)\s*[:=]?\s*(\d{2,3}(?:\.\d)?)\s*(°?[cf])?", raw_text)
    if temp_match:
        try:
            val = float(temp_match.group(1))
            unit = temp_match.group(2)
            if unit and "c" in unit.lower():
                result["temp_unit"] = "C"
            result["temperature"] = val
        except ValueError:
            pass

    # 5. Age Pattern (e.g., Age: 45y, Age: 32, 58 M, 24 F, 32M)
    age_match = re.search(r"(?i)(?:age|yrs?)\s*[:=]?\s*(\d{1,3})", raw_text)
    if not age_match:
        # Clinical shorthand e.g. "32 M", "45 F", "24yo"
        age_match = re.search(r"\b(\d{1,3})\s*(?:yo|y/o|y|[MF])\b", raw_text, re.IGNORECASE)
    if age_match:
        try:
            val = int(age_match.group(1))
            if 0 <= val <= 120:
                result["age"] = val
        except ValueError:
            pass

    # 6. Gender Pattern (e.g., Male, Female, M, F)
    if re.search(r"(?i)\b(?:female|woman|f)\b", raw_text):
        result["gender"] = "Female"
    elif re.search(r"(?i)\b(?:male|man|m)\b", raw_text):
        result["gender"] = "Male"

    # 7. Complaint extraction (e.g., Complaint: ..., Symptoms: ...)
    comp_match = re.search(r"(?i)(?:complaint|symptoms?|c/o|presenting)\s*[:=]?\s*([^\n\r]+)", raw_text)
    if comp_match:
        result["complaint"] = comp_match.group(1).strip()
    else:
        # Fallback to remaining lines with symptom keywords
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        for line in lines:
            if any(k in line.lower() for k in ["pain", "fever", "cough", "breath", "vomit", "chest", "dard", "noppi"]):
                result["complaint"] = line
                break

    return result


def process_slip_image(image_bytes: bytes) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Processes an uploaded or camera-captured slip image completely in-memory.
    Never writes the image to disk.
    Returns: (success: bool, message: str, parsed_fields: dict)
    """
    is_avail, status_msg = check_ocr_status()

    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
    except Exception as e:
        return False, f"Invalid image format: {str(e)}", {}

    if not is_avail:
        return False, f"OCR engine: not installed ({status_msg}). Offline Tesseract required.", {}

    try:
        raw_text = pytesseract.image_to_string(image)
        parsed = parse_clinical_text(raw_text)
        return True, "Slip parsed successfully via offline OCR", parsed
    except Exception as e:
        return False, f"OCR processing failed: {str(e)}", {}


# Curated sample clinical slips for rapid demonstration / testing
SAMPLE_SLIPS = {
    "Sample 1: Hypoxia & Chest Pain": (
        "GOVT GENERAL HOSPITAL OPD\n"
        "TRIAGE INITIAL SLIP\n"
        "Age: 56 Y  Gender: Male\n"
        "SpO2: 89%\n"
        "BP: 165/95 mmHg\n"
        "Pulse: 112 bpm\n"
        "Temp: 99.1 F\n"
        "Complaint: Acute severe chest pain radiating to left arm and breathlessness"
    ),
    "Sample 2: Hypertensive Crisis": (
        "DISTRICT HEADQUARTERS HOSPITAL\n"
        "Age: 62  Gender: Female\n"
        "BP: 195/125\n"
        "SpO2: 96%\n"
        "Pulse: 98\n"
        "Temp: 98.4 F\n"
        "Complaint: Sudden blinding headache, dizziness and blurred vision"
    ),
    "Sample 3: High Fever & Cough": (
        "COMMUNITY HEALTH CENTRE\n"
        "Age: 28  Gender: Female\n"
        "SpO2: 97%\n"
        "BP: 115/75\n"
        "Pulse: 104 bpm\n"
        "Temp: 102.8 F\n"
        "Complaint: High fever for 3 days with severe productive cough and body chills"
    )
}


def render_ai_scan_widget():
    """
    Renders the AI Scan (Offline OCR) widget inside st.expander with:
    - Status badge: 'OCR engine: available / not installed'
    - st.camera_input and st.file_uploader
    - Per-field review table with checkboxes
    - 'Apply selected to form' button filling only ticked values
    - In-memory processing only (never stored to disk)
    """
    with st.expander("📷 AI Scan: Paper Triage Slip (Offline OCR)", expanded=False):
        is_avail, ocr_msg = check_ocr_status()

        # 1. OCR Engine Status Badge
        if is_avail:
            render_html(
                f'<div class="status-pill" style="border-color: rgba(0,214,143,0.4); color: #66E3C4; margin-bottom: 12px; display: inline-flex;">'
                f'<span class="status-dot status-dot-green"></span>'
                f'<span>OCR engine: available ({ocr_msg})</span>'
                f'</div>'
            )
        else:
            render_html(
                '<div class="status-pill" style="border-color: rgba(255,199,0,0.4); color: #FFD13B; margin-bottom: 8px; display: inline-flex;">'
                '<span class="status-dot status-dot-orange"></span>'
                '<span>OCR engine: not installed</span>'
                '</div>'
                '<div style="font-size: 0.80rem; color: #94A3B8; margin-bottom: 12px; line-height: 1.4;">'
                'Offline Tesseract OCR binary not detected on system PATH. '
                'Paper slips can be processed once installed (see README.md). '
                'You can test the extraction pipeline using the sample paper slips below.'
                '</div>'
            )

        st.caption("🔒 Privacy & Clinical Safety: Images are processed exclusively in RAM and never written to disk. No cloud APIs.")

        # 2. Input Methods
        tab_cam, tab_file, tab_samples = st.tabs([
            "📸 Camera Snap",
            "📁 File Upload",
            "📝 Test Sample Slips"
        ])

        new_parsed = None

        with tab_cam:
            cam_file = st.camera_input("Capture photo of paper triage slip", key="ai_scan_cam_input")
            if cam_file is not None:
                img_bytes = cam_file.getvalue()
                success, msg, parsed = process_slip_image(img_bytes)
                if success:
                    st.success(f"✅ {msg}")
                    new_parsed = parsed
                else:
                    st.warning(f"⚠️ {msg}")

        with tab_file:
            up_file = st.file_uploader(
                "Upload slip image (PNG, JPG, JPEG)",
                type=["png", "jpg", "jpeg"],
                key="ai_scan_file_uploader"
            )
            if up_file is not None:
                img_bytes = up_file.getvalue()
                success, msg, parsed = process_slip_image(img_bytes)
                if success:
                    st.success(f"✅ {msg}")
                    new_parsed = parsed
                else:
                    st.warning(f"⚠️ {msg}")

        with tab_samples:
            st.write("Select a representative paper slip to test extraction:")
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                if st.button("Sample 1: Hypoxia", width="stretch"):
                    new_parsed = parse_clinical_text(SAMPLE_SLIPS["Sample 1: Hypoxia & Chest Pain"])
                    st.info("Loaded Sample 1 text!")
            with col_s2:
                if st.button("Sample 2: HTN Crisis", width="stretch"):
                    new_parsed = parse_clinical_text(SAMPLE_SLIPS["Sample 2: Hypertensive Crisis"])
                    st.info("Loaded Sample 2 text!")
            with col_s3:
                if st.button("Sample 3: High Fever", width="stretch"):
                    new_parsed = parse_clinical_text(SAMPLE_SLIPS["Sample 3: High Fever & Cough"])
                    st.info("Loaded Sample 3 text!")

        if new_parsed:
            st.session_state["ocr_parsed_fields"] = new_parsed

        # 3. Review Table with a Checkbox Per Field
        if "ocr_parsed_fields" in st.session_state:
            parsed = st.session_state["ocr_parsed_fields"]
            st.markdown("---")
            st.markdown("### 📋 Review Extracted Clinical Fields")
            st.caption("Check or uncheck individual fields before applying to the intake form. Only ticked values are filled.")

            col_r1, col_r2 = st.columns(2)

            field_selections = {}

            with col_r1:
                # Age
                age_val = parsed.get("age")
                age_checked = st.checkbox(
                    f"Age: **{age_val if age_val is not None else 'Not detected'}**",
                    value=(age_val is not None),
                    key="ocr_field_age"
                )
                if age_checked and age_val is not None:
                    field_selections["age"] = age_val

                # Gender
                gender_val = parsed.get("gender")
                gender_checked = st.checkbox(
                    f"Gender: **{gender_val if gender_val else 'Not detected'}**",
                    value=(gender_val is not None),
                    key="ocr_field_gender"
                )
                if gender_checked and gender_val:
                    field_selections["gender"] = gender_val

                # SpO2
                spo2_val = parsed.get("spo2")
                spo2_checked = st.checkbox(
                    f"SpO2: **{f'{spo2_val}%' if spo2_val is not None else 'Not detected'}**",
                    value=(spo2_val is not None),
                    key="ocr_field_spo2"
                )
                if spo2_checked and spo2_val is not None:
                    field_selections["spo2"] = spo2_val

                # Blood Pressure
                sbp_val = parsed.get("bp_systolic")
                dbp_val = parsed.get("bp_diastolic")
                bp_display = f"{sbp_val}/{dbp_val} mmHg" if (sbp_val and dbp_val) else "Not detected"
                bp_checked = st.checkbox(
                    f"Blood Pressure: **{bp_display}**",
                    value=(sbp_val is not None and dbp_val is not None),
                    key="ocr_field_bp"
                )
                if bp_checked and sbp_val is not None:
                    field_selections["bp_systolic"] = sbp_val
                    field_selections["bp_diastolic"] = dbp_val

            with col_r2:
                # Pulse
                pulse_val = parsed.get("pulse")
                pulse_checked = st.checkbox(
                    f"Pulse Rate: **{f'{pulse_val} bpm' if pulse_val is not None else 'Not detected'}**",
                    value=(pulse_val is not None),
                    key="ocr_field_pulse"
                )
                if pulse_checked and pulse_val is not None:
                    field_selections["pulse"] = pulse_val

                # Temperature
                temp_val = parsed.get("temperature")
                t_unit = parsed.get("temp_unit", "F")
                temp_display = f"{temp_val}°{t_unit}" if temp_val is not None else "Not detected"
                temp_checked = st.checkbox(
                    f"Temperature: **{temp_display}**",
                    value=(temp_val is not None),
                    key="ocr_field_temp"
                )
                if temp_checked and temp_val is not None:
                    field_selections["temperature"] = temp_val
                    field_selections["temp_unit"] = t_unit

                # Chief Complaint
                comp_val = parsed.get("complaint", "")
                comp_display = comp_val if comp_val else "Not detected"
                comp_checked = st.checkbox(
                    f"Chief Complaint: **{comp_display[:60]}...**" if len(comp_display) > 60 else f"Chief Complaint: **{comp_display}**",
                    value=bool(comp_val),
                    key="ocr_field_complaint"
                )
                if comp_checked and comp_val:
                    field_selections["complaint"] = comp_val

            # 4. Action: Apply selected to form
            if st.button("📥 Apply selected to form", type="primary", width="stretch"):
                if not field_selections:
                    st.warning("No fields are currently selected. Tick at least one field to apply.")
                else:
                    if "intake_draft" not in st.session_state:
                        st.session_state["intake_draft"] = {}
                    st.session_state["intake_draft"].update(field_selections)
                    st.success(f"Applied {len(field_selections)} selected field(s) to intake form draft below!")
                    st.rerun()
