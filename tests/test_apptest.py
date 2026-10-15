"""CarePriority Gov - Streamlit AppTest Suite
Verifies that all 7 navigation pages render without unhandled exceptions.
"""

import os
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))

PAGES = [
    "📊 Command Center",
    "⚡ Patient Intake (<60s)",
    "📋 Live Doctor OPD Queue",
    "📺 Public Status Display",
    "🔍 Encounter Detail View",
    "🛡️ Safety Rules & Thresholds",
    "🧪 Live Evaluation & Benchmarks"
]

@pytest.mark.parametrize("page_name", PAGES)
def test_page_renders_without_exception(page_name):
    """Verifies that each page in app.py renders cleanly without exceptions."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception, f"Initial boot failed before selecting {page_name}: {at.exception}"
    
    if at.sidebar.radio and len(at.sidebar.radio) > 0:
        at.sidebar.radio[0].set_value(page_name)
        at.run()
        assert not at.exception, f"Exception on page '{page_name}': {at.exception}"
