"""Tests for the additive helpers on the original Patient/Symptom agents."""

from app.agents.patient_agent import PatientDataAgent
from app.agents.symptom_agent import SymptomAnalysisAgent, duration_category, parse_duration_days
from app.models.schemas import PatientCase


def test_parse_duration():
    assert parse_duration_days("5 days") == 5
    assert parse_duration_days("3 weeks") == 21
    assert parse_duration_days("2 months") == 60
    assert parse_duration_days("a while") is None
    assert parse_duration_days(None) is None
    assert duration_category(5) == "short (up to 2 weeks)"
    assert duration_category(120) == "long-standing (over 3 months)"
    assert parse_duration_days("2 hours") == 0
    assert parse_duration_days("45 minutes") == 0
    assert duration_category(parse_duration_days("2 hours")) == "very recent (less than a day)"


def test_symptom_red_flags_and_systems():
    patient = PatientCase(patient_id="SIM-R", age=58, sex="Male",
                          symptoms=["Chest pain", "sweating", "Shortness of breath"])
    result = SymptomAnalysisAgent().analyze(patient)
    assert result["red_flags"] == ["chest pain", "shortness of breath"]
    assert "cardiovascular" in result["body_systems"]
    assert "respiratory" in result["body_systems"]


def test_process_raw_valid_and_invalid():
    agent = PatientDataAgent()
    ok = agent.process_raw({"patient_id": "SIM-1", "age": 30, "sex": "Female", "symptoms": ["cough"]})
    assert ok["validation"]["valid"] is True
    assert isinstance(ok["patient"], PatientCase)

    bad = agent.process_raw({"patient_id": "SIM-2", "age": -3, "sex": "Female",
                             "symptoms": ["cough"], "lab_results": {"WBC": "abc"}})
    assert bad["patient"] is None
    assert bad["validation"]["valid"] is False
    assert len(bad["validation"]["issues"]) == 2

    assert agent.process_raw("not a dict")["validation"]["valid"] is False
