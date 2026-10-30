"""CarePriority Gov - HTML Helper Rendering Tests
==============================================
Validates that all HTML helper outputs (explainability, smart_ticket, surge_mode, styles)
are strictly free of 4+ space indentations before tags and have no blank lines between tags.
This prevents CommonMark from parsing HTML as indented code blocks.
"""

import re
import pytest
from datetime import datetime
from typing import Dict, Any

import explainability
import smart_ticket
import surge_mode
import styles


def make_encounter(
    tier: int | None,
    name: str,
    complaint: str,
    vitals: Dict[str, Any],
    abstained: bool = False,
    abstention_reason: str | None = None,
    rule_triggered: str | None = None,
    red_flags: list | None = None,
    model_probs: list | None = None
) -> Dict[str, Any]:
    return {
        "encounter_id": f"ENC-TEST-T{tier or 0}",
        "created_at": datetime.now().isoformat(),
        "patient_name": name,
        "age": 45,
        "gender": "Female",
        "department": "Emergency Medicine" if (tier in [1, 2]) else "General Medicine",
        "tier": tier,
        "tier_name": f"Level {tier}" if tier else "Needs Nurse Review",
        "abstained": abstained,
        "abstention_reason": abstention_reason,
        "is_escalated": bool(rule_triggered),
        "rule_triggered": rule_triggered,
        "reasons": [f"Triage reason for {name}"],
        "red_flags": red_flags or [],
        "complaint_original": complaint,
        "vitals": vitals,
        "model_tier": tier or 3,
        "model_confidence": 0.90 if tier else None,
        "model_probs": model_probs if model_probs is not None else ([0.2] * 5 if tier else None),
        "status": "Awaiting Nurse Review" if abstained else "Waiting",
        "audit": [
            {
                "timestamp": datetime.now().isoformat(),
                "actor": "Triage Nurse",
                "action": "Intake Submission",
                "tier": tier,
                "department": "General Medicine",
                "reason": "Initial rapid intake assessment"
            }
        ]
    }


TEST_ENCOUNTERS = {
    "L1": make_encounter(
        tier=1,
        name="Critical Patient L1",
        complaint="Severe crushing chest pain radiating to jaw and diaphoresis",
        vitals={"spo2": 82.0, "bp_systolic": 75, "bp_diastolic": 45, "pulse": 138, "temperature": 98.6},
        rule_triggered="SpO2 Critical (< 85%)",
        red_flags=["chest pain", "severe breathlessness"],
        model_probs=[0.88, 0.08, 0.02, 0.01, 0.01]
    ),
    "L2": make_encounter(
        tier=2,
        name="Emergent Patient L2",
        complaint="High fever and severe shortness of breath",
        vitals={"spo2": 89.5, "bp_systolic": 115, "bp_diastolic": 75, "pulse": 112, "temperature": 103.1},
        rule_triggered="SpO2 Hypoxia (< 92%)",
        red_flags=["high fever", "breathing difficulty"],
        model_probs=[0.05, 0.82, 0.08, 0.03, 0.02]
    ),
    "L3": make_encounter(
        tier=3,
        name="Urgent Patient L3",
        complaint="Persistent acute lower abdominal pain for 6 hours",
        vitals={"spo2": 98.0, "bp_systolic": 128, "bp_diastolic": 82, "pulse": 86, "temperature": 99.0},
        model_probs=[0.02, 0.06, 0.78, 0.10, 0.04]
    ),
    "L4": make_encounter(
        tier=4,
        name="Less Urgent Patient L4",
        complaint="Mild ankle swelling after twisted step 2 days ago",
        vitals={"spo2": 99.0, "bp_systolic": 120, "bp_diastolic": 78, "pulse": 72, "temperature": 98.4},
        model_probs=[0.01, 0.02, 0.08, 0.82, 0.07]
    ),
    "L5": make_encounter(
        tier=5,
        name="Non Urgent Patient L5",
        complaint="Routine monthly refill of diabetes medication",
        vitals={"spo2": 98.0, "bp_systolic": 122, "bp_diastolic": 76, "pulse": 68, "temperature": 98.2},
        model_probs=[0.0, 0.01, 0.03, 0.05, 0.91]
    ),
    "Abstained": make_encounter(
        tier=None,
        name="Abstained Walkin",
        complaint="Feeling unwell",
        vitals={"spo2": None, "bp_systolic": None, "bp_diastolic": None, "pulse": None, "temperature": None},
        abstained=True,
        abstention_reason="Ambiguous complaint without clinical vitals",
        model_probs=None
    )
}


def validate_html_output(html_content: str, label: str):
    """
    Asserts that:
    1. Output is non-empty string.
    2. No line begins with 4+ spaces followed by '<'.
    3. No blank line exists inside or between HTML tags.
    """
    assert isinstance(html_content, str), f"{label} must return str"
    assert len(html_content.strip()) > 0, f"{label} returned empty HTML"

    lines = html_content.splitlines()
    for idx, line in enumerate(lines, 1):
        # Check: 4+ spaces indentation before a tag
        assert not re.match(r"^\s{4,}<", line), (
            f"HTML indentation bug in {label} at line {idx}: {line!r}"
        )
        # Check: no blank line
        assert line.strip() != "", (
            f"Blank line detected in {label} at line {idx}"
        )

    # Double newline check
    assert "\n\n" not in html_content, f"Consecutive newlines detected in {label}"


@pytest.mark.parametrize("level_name, encounter", list(TEST_ENCOUNTERS.items()))
def test_explainability_horizontal_view_html(level_name, encounter):
    """Tests explainability card for L1-L5 and abstained encounters."""
    html_markup = explainability.render_horizontal_contribution_view(encounter)
    validate_html_output(html_markup, f"explainability.render_horizontal_contribution_view({level_name})")


@pytest.mark.parametrize("level_name, encounter", list(TEST_ENCOUNTERS.items()))
def test_smart_token_ticket_patient_facing_html(level_name, encounter):
    """Tests patient-facing Smart Token Ticket for L1-L5 and abstained encounters."""
    html_markup = smart_ticket.render_smart_token_ticket(
        encounter, queue_pos=2, est_wait_mins=20, show_staff_clinical_info=False
    )
    validate_html_output(html_markup, f"smart_ticket.patient_ticket({level_name})")


@pytest.mark.parametrize("level_name, encounter", list(TEST_ENCOUNTERS.items()))
def test_smart_token_ticket_staff_chart_html(level_name, encounter):
    """Tests staff-facing clinical chart Smart Token Ticket for L1-L5 and abstained encounters."""
    html_markup = smart_ticket.render_smart_token_ticket(
        encounter, queue_pos=2, est_wait_mins=20, show_staff_clinical_info=True
    )
    validate_html_output(html_markup, f"smart_ticket.staff_ticket({level_name})")


@pytest.mark.parametrize("level_name, encounter", list(TEST_ENCOUNTERS.items()))
def test_tier_badge_html(level_name, encounter):
    """Tests tier badge HTML for L1-L5 and abstained encounters."""
    html_markup = styles.render_tier_badge(encounter.get("tier"), encounter.get("abstained", False))
    validate_html_output(html_markup, f"styles.render_tier_badge({level_name})")


def test_surge_mode_banner_html():
    """Tests hospital surge mode banner HTML generation."""
    # Normal status returns empty string
    normal_status = {"is_active": False}
    assert surge_mode.render_surge_banner_html(normal_status) == ""

    # Amber surge
    amber_status = {
        "is_active": True,
        "severity": "LEVEL 1 AMBER SURGE",
        "waiting_count": 18,
        "emergent_count": 4,
        "waiting_threshold": 15,
        "emergent_threshold": 3,
        "reasons": ["Active OPD queue volume reached 18", "High acuity patients reached 4"]
    }
    amber_html = surge_mode.render_surge_banner_html(amber_status)
    validate_html_output(amber_html, "surge_mode.render_surge_banner_html(Amber)")

    # Red surge
    red_status = {
        "is_active": True,
        "severity": "LEVEL 2 RED SURGE",
        "waiting_count": 35,
        "emergent_count": 10,
        "waiting_threshold": 15,
        "emergent_threshold": 3,
        "reasons": ["Extreme overcrowding", "Critical emergent surge"]
    }
    red_html = surge_mode.render_surge_banner_html(red_status)
    validate_html_output(red_html, "surge_mode.render_surge_banner_html(Red)")
