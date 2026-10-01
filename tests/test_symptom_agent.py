from app.agents.symptom_agent import SymptomAnalysisAgent
from app.models.schemas import PatientCase


def test_symptom_analysis():

    patient = PatientCase(
        patient_id="SIM-001",
        age=45,
        sex="Male",
        symptoms=["fever", "cough", "fatigue"],
        symptom_duration="5 days",
        medical_history=["hypertension"],
        medications=["antihypertensive medication"],
        allergies=[],
        lab_results={
            "WBC": 13500,
            "CRP": 48
        }
    )

    agent = SymptomAnalysisAgent()

    result = agent.analyze(patient)

    assert result["patient_id"] == "SIM-001"
    assert result["symptom_count"] == 3
    assert "fever" in result["key_features"]
    assert result["duration"] == "5 days"


def test_no_symptoms():

    patient = PatientCase(
        patient_id="SIM-002",
        age=30,
        sex="Female",
        symptoms=[],
        medical_history=[],
        medications=[],
        allergies=[],
        lab_results={}
    )

    agent = SymptomAnalysisAgent()

    result = agent.analyze(patient)

    assert result["symptom_count"] == 0
    assert result["key_features"] == []