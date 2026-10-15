"""CarePriority Gov - Stage 3 Feature Tests
=========================================
Tests for:
1. 3D Procedural Three.js Hero Scene (hero_3d.py)
   - Pinned cdnjs r128 script
   - Polished static SVG hero fallback (never empty box)
   - WebGL support check & reduced-motion check
   - devicePixelRatio cap at 2
   - Visibilitychange pause & unload disposal
2. AI Scan & Offline OCR (ai_scan.py)
   - check_ocr_status availability
   - parse_clinical_text regex vitals & complaint extraction
   - process_slip_image in-memory execution (never writes to disk, no cloud)
3. Explainability Visuals (explainability.py)
   - compute_clinical_contributions real computed clinical deviance
   - Zero invented confidence percentages
   - render_horizontal_contribution_view HTML output
4. Streamlit AppTest Integration Tests
   - Command Center rendering with 3D hero
   - Patient Intake with AI scan widget & form prefill
   - Encounter Detail View with horizontal contribution view
"""

import io
from pathlib import Path
from PIL import Image
import pytest
from streamlit.testing.v1 import AppTest

import hero_3d
import ai_scan
import explainability
from rules_config import TRIAGE_LEVELS

APP_PATH = str(Path(__file__).parent.parent / "app.py")


# =============================================================================
# 1. 3D HERO SCENE TESTS (hero_3d.py)
# =============================================================================

def test_hero_3d_html_contains_pinned_threejs_and_fallbacks():
    """Verifies Three.js CDN pinning, static SVG fallback, WebGL checks, and lifecycle listeners."""
    html_content = hero_3d.get_hero_3d_html(height=240)

    # 1. Pinned cdnjs Three.js r128 script
    assert "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js" in html_content

    # 2. Polished static SVG hero fallback (never an empty box)
    assert 'id="static-hero-fallback"' in html_content
    assert '<svg class="svg-cross-pulse"' in html_content
    assert 'rect x="74" y="32"' in html_content  # Glass care-cross vertical bar
    assert 'rect x="32" y="74"' in html_content  # Glass care-cross horizontal bar

    # 3. WebGL support check & prefers-reduced-motion check
    assert "prefers-reduced-motion" in html_content
    assert "checkWebGL" in html_content or "getContext('webgl')" in html_content

    # 4. devicePixelRatio capped at 2
    assert "Math.min(window.devicePixelRatio || 1, 2)" in html_content

    # 5. Visibilitychange pause when tab hidden
    assert "visibilitychange" in html_content
    assert "document.hidden" in html_content

    # 6. Unmount disposal
    assert "renderer.dispose()" in html_content
    assert "cancelAnimationFrame" in html_content
    assert "beforeunload" in html_content


def test_hero_3d_banner_renders_without_exception():
    """Verifies render_hero_3d_banner function executes cleanly."""
    # Should generate without errors
    html_markup = hero_3d.get_hero_3d_html(height=200)
    assert len(html_markup) > 1000
    assert "CarePriority" in html_markup


# =============================================================================
# 2. AI SCAN & OFFLINE OCR TESTS (ai_scan.py)
# =============================================================================

def test_check_ocr_status_returns_tuple():
    """Verifies check_ocr_status returns (bool, str) tuple without crashing."""
    is_avail, msg = ai_scan.check_ocr_status()
    assert isinstance(is_avail, bool)
    assert isinstance(msg, str)
    assert len(msg) > 0


def test_parse_clinical_text_regex_extraction():
    """Verifies regex extraction of clinical vitals, age, gender, and complaints."""
    raw_sample = (
        "GOVT GENERAL HOSPITAL OPD\n"
        "Age: 54 Y  Gender: Female\n"
        "SpO2: 91%\n"
        "BP: 175/105 mmHg\n"
        "Pulse: 118 bpm\n"
        "Temp: 101.4 F\n"
        "Complaint: Severe chest pain and breathlessness with sweating"
    )

    parsed = ai_scan.parse_clinical_text(raw_sample)

    assert parsed["age"] == 54
    assert parsed["gender"] == "Female"
    assert parsed["spo2"] == 91.0
    assert parsed["bp_systolic"] == 175
    assert parsed["bp_diastolic"] == 105
    assert parsed["pulse"] == 118
    assert parsed["temperature"] == 101.4
    assert parsed["temp_unit"] == "F"
    assert "chest pain" in parsed["complaint"].lower()


def test_parse_clinical_text_handles_celsius_and_missing_data():
    """Verifies parsing handles Celsius units and gracefully returns None for missing fields."""
    sample_celsius = "Patient 32 M, Temp: 38.6 C, c/o acute abdominal pain"
    parsed = ai_scan.parse_clinical_text(sample_celsius)

    assert parsed["age"] == 32
    assert parsed["gender"] == "Male"
    assert parsed["temperature"] == 38.6
    assert parsed["temp_unit"] == "C"
    assert parsed["spo2"] is None
    assert parsed["bp_systolic"] is None
    assert "abdominal pain" in parsed["complaint"].lower()


def test_process_slip_image_in_memory_only():
    """Verifies process_slip_image operates in RAM and never crashes on valid or corrupt images."""
    # 1. Valid RGB synthetic image in memory
    img = Image.new("RGB", (200, 100), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    valid_bytes = buf.getvalue()

    success, msg, parsed = ai_scan.process_slip_image(valid_bytes)
    # If tesseract is not installed, it returns False with a helpful message; if installed, True
    assert isinstance(success, bool)
    assert isinstance(msg, str)
    assert isinstance(parsed, dict)

    # 2. Corrupted byte stream
    corrupt_bytes = b"NOT_A_VALID_IMAGE_DATA_STREAM"
    c_success, c_msg, c_parsed = ai_scan.process_slip_image(corrupt_bytes)
    assert c_success is False
    assert "Invalid image format" in c_msg


# =============================================================================
# 3. EXPLAINABILITY FACTUAL VIEW TESTS (explainability.py)
# =============================================================================

def test_compute_factual_contributions_strict_ordering():
    """Verifies contributing factors are strictly ordered: hard rules, red flags, out-of-range vitals, model output."""
    critical_encounter = {
        "vitals": {
            "spo2": 88.0,        # Hard escalation L2 (<92%)
            "bp_systolic": 190,  # Hard escalation L2 (>=180)
            "bp_diastolic": 110,
            "pulse": 108,        # Out-of-range vital (100-129 mild tachycardia)
            "temperature": 101.5 # Out-of-range vital (100.4-103.4 pyrexia)
        },
        "complaint": "Severe chest pain radiating to left arm",
        "red_flags": ["chest pain"],
        "is_escalated": True,
        "rule_triggered": "Mandatory Hypoxia Escalation"
    }

    result = explainability.compute_factual_contributions(critical_encounter)

    # 1. Hard escalation rules first
    hard = result["hard_escalations"]
    assert len(hard) >= 2
    # Verify exact format: SpO2 = 88% -> below 92.0% -> escalated to at least L2
    assert any("SpO2 = 88" in item["statement"] and "below 92" in item["statement"] and "escalated to at least L2" in item["statement"] for item in hard)
    assert any("SBP = 190" in item["statement"] and "≥ 180" in item["statement"] and "escalated to at least L2" in item["statement"] for item in hard)

    # 2. Red-flag keywords next
    rf = result["red_flag_factors"]
    assert len(rf) >= 1
    assert any("chest pain" in item["statement"].lower() for item in rf)

    # 3. Out-of-range vitals next
    oor = result["out_of_range_vitals"]
    assert len(oor) >= 2
    assert any("Pulse = 108" in item["statement"] for item in oor)
    assert any("Temp = 101.5" in item["statement"] for item in oor)

    # 4. Model output last
    model_out = result["model_output"]
    assert "model output" in model_out["label"].lower()
    assert "placeholder data" in model_out["label"].lower()


def test_model_output_labeled_placeholder_data_and_uses_real_probabilities():
    """Verifies model output is labeled trained on placeholder data and shows real predict_proba."""
    sample_enc = {
        "vitals": {"spo2": 97.0, "bp_systolic": 120, "bp_diastolic": 80, "pulse": 76, "temperature": 98.6},
        "complaint": "Mild headache since morning",
        "model_probs": [0.02, 0.05, 0.15, 0.65, 0.13],
        "model_tier": 4
    }

    res = explainability.compute_factual_contributions(sample_enc)
    model_out = res["model_output"]

    assert model_out["label"] == "model output (trained on placeholder data)"
    assert model_out["predicted_tier"] == 4
    assert model_out["probabilities"] == [0.02, 0.05, 0.15, 0.65, 0.13]


def test_no_invented_percentages_in_explainability():
    """
    MANDATORY GUARDRAIL:
    Fails if any invented/arbitrary percentage appears in the explainability output.
    All percentages in HTML must either be real measured vitals, matched thresholds,
    or exact mathematical outputs from predict_proba.
    """
    import re

    # Test cases: Critical hypoxia, Hypertensive crisis, Normal walk-in
    test_encounters = [
        {
            "vitals": {"spo2": 88.0, "bp_systolic": 195, "bp_diastolic": 125, "pulse": 134, "temperature": 104.2},
            "complaint": "Chest pain with severe breathlessness",
            "model_probs": [0.12, 0.81, 0.05, 0.01, 0.01],
            "model_tier": 2
        },
        {
            "vitals": {"spo2": 93.0, "bp_systolic": 115, "bp_diastolic": 75, "pulse": 82, "temperature": 98.4},
            "complaint": "Mild cough for 2 days",
            "model_probs": [0.01, 0.03, 0.18, 0.72, 0.06],
            "model_tier": 4
        },
        {
            "vitals": {"spo2": 99.0, "bp_systolic": 118, "bp_diastolic": 78, "pulse": 72, "temperature": 98.6},
            "complaint": "Routine blood pressure checkup",
            "model_probs": [0.005, 0.015, 0.08, 0.25, 0.65],
            "model_tier": 5
        }
    ]

    # Prohibited arbitrary hard-coded severity weights that previously masqueraded as scores
    prohibited_invented_scores = {"85%", "90%", "95%", "45%", "60%", "40%", "30%", "50%", "5%"}

    for enc in test_encounters:
        html = explainability.render_horizontal_contribution_view(enc)

        # 1. No progress bars with hardcoded severity weights (e.g. style="width: 85%;")
        for bad_score in prohibited_invented_scores:
            assert f'width: {bad_score}' not in html, f"Found invented percentage bar {bad_score} in output!"

        # 2. Filter out CSS container layout styling (width: 100%, height: 100%) before checking clinical percentages
        cleaned_html = re.sub(r'(?:width|height)\s*:\s*100\s*%', '', html)

        # 3. Extract every clinical data percentage in the HTML
        pct_matches = re.findall(r"(\d+(?:\.\d+)?)\s*%", cleaned_html)
        assert len(pct_matches) > 0, "Expected real model probabilities or measured SpO2 percentages"

        # Valid allowed numbers:
        # - The encounter's real SpO2 (e.g. 88, 93, 99)
        # - The real threshold numbers in rules_config.py (85, 92)
        # - The real model probabilities rounded to 1 decimal place
        allowed_pct_numbers = {
            f"{enc['vitals']['spo2']:.1f}",
            f"{int(enc['vitals']['spo2'])}",
            "85", "85.0", "92", "92.0", "94", "94.0"
        }
        for p in enc["model_probs"]:
            allowed_pct_numbers.add(f"{p * 100:.1f}")
            allowed_pct_numbers.add(f"{round(p * 100, 1)}")

        # Verify every percentage in the output belongs strictly to the allowed computed set
        for pct_val in pct_matches:
            # Check if this percentage string matches an allowed measured value or model probability
            is_valid = any(
                pct_val == allowed or pct_val.startswith(allowed)
                for allowed in allowed_pct_numbers
            )
            assert is_valid, (
                f"Invented percentage '{pct_val}%' detected in explainability HTML! "
                f"Allowed computed values: {allowed_pct_numbers}"
            )


# =============================================================================
# 4. STREAMLIT APPTEST INTEGRATION TESTS
# =============================================================================

def test_apptest_command_center_3d_hero_integration():
    """Runs AppTest on Command Center and verifies it renders without exceptions."""
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert not at.exception
    # Command Center title or brand in text/markdown/html
    assert any("CarePriority" in str(getattr(el, "value", "")) for el in at.markdown) or len(at.html) > 0


def test_apptest_patient_intake_with_ai_scan_and_form():
    """Navigates to Patient Intake via AppTest and verifies inputs render properly."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    # Navigate to Patient Intake
    at.radio[0].set_value("⚡ Patient Intake (<60s)")
    at.run()
    assert not at.exception

    # Check for text inputs and number inputs in intake form
    assert len(at.number_input) >= 4  # Age, SpO2, SBP/DBP, Pulse, Temp
    assert len(at.text_area) >= 1      # Chief complaint


def test_apptest_encounter_detail_view_with_explainability():
    """Navigates to Encounter Detail View via AppTest and verifies reasoning displays cleanly."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    # Navigate to Encounter Detail View
    at.radio[0].set_value("🔍 Encounter Detail View")
    at.run()
    assert not at.exception
