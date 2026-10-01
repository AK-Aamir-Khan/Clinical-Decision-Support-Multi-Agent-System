import math

import pytest

from app.agents.lab_agent import LaboratoryAnalysisAgent
from app.models.schemas import PatientCase


@pytest.fixture(scope="module")
def agent():
    return LaboratoryAnalysisAgent()


def test_high_values_detected(agent, respiratory_case):
    result = agent.analyze(respiratory_case)
    statuses = {r["test"]: r["status"] for r in result["results"]}

    assert statuses == {"WBC": "high", "CRP": "high"}
    assert len(result["abnormal_findings"]) == 2
    assert "not a diagnosis" in result["note"]


def test_alias_and_normal_and_low(agent):
    assert agent.evaluate_value("hb", 14.0, "Male")["status"] == "normal"
    assert agent.evaluate_value("Haemoglobin", 9.5, "Female")["status"] == "low"
    assert agent.evaluate_value("platelet count", 90000)["test"] == "Platelets"


def test_sex_specific_range(agent):
    # 13.0 g/dL is below the male range but inside the female range
    assert agent.evaluate_value("Hemoglobin", 13.0, "Male")["status"] == "low"
    assert agent.evaluate_value("Hemoglobin", 13.0, "Female")["status"] == "normal"


def test_invalid_and_unknown_values(agent):
    patient = PatientCase(
        patient_id="SIM-L",
        age=40,
        sex="Female",
        symptoms=["fatigue"],
        lab_results={"WBC": -5, "CRP": math.nan, "Mystery marker": 3.2},
    )
    result = agent.analyze(patient)
    statuses = {r["input_name"]: r["status"] for r in result["results"]}

    assert statuses["WBC"] == "invalid"
    assert statuses["CRP"] == "invalid"
    assert statuses["Mystery marker"] == "unknown"
    assert set(result["invalid_values"]) == {"WBC", "CRP"}
    assert result["abnormal_findings"] == []


def test_no_labs(agent, empty_history_case):
    result = agent.analyze(empty_history_case)
    assert result["results"] == []
    assert result["summary"] == "No laboratory results were supplied."
