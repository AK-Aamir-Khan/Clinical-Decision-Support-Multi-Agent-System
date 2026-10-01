"""Headless UI smoke tests using Streamlit's AppTest (no browser, no API key)."""

from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("USE_LLM", "false")
    at = AppTest.from_file(str(MAIN), default_timeout=60)
    at.run()
    return at


def test_app_loads_with_disclaimer(app):
    assert not app.exception
    assert any("Educational simulation only" in e.value for e in app.error)
    assert "Select a synthetic case" in app.info[0].value


def test_run_default_case(app):
    app.button[0].click().run()
    assert not app.exception
    subheaders = [s.value for s in app.subheader]
    assert subheaders[0] == "1. Patient information"
    assert "9. Safety review" in subheaders
    assert any("Community-acquired pneumonia" in m.value for m in app.markdown)


def test_run_invalid_case(app):
    app.selectbox[0].select_index(len(app.selectbox[0].options) - 1).run()
    app.button[0].click().run()
    assert not app.exception
    assert any("invalid" in e.value for e in app.error)


def test_malformed_json_is_reported(app):
    app.text_area[0].input("{not json").run()
    app.button[0].click().run()
    assert not app.exception
    assert any("Invalid JSON" in e.value for e in app.error)
