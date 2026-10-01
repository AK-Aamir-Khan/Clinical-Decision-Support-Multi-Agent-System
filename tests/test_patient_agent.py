from app.agents.patient_agent import PatientDataAgent
from app.models.schemas import PatientCase


def test_patient_agent_valid_case():

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

    agent = PatientDataAgent()

    result = agent.process_patient(patient)

    assert result["patient_id"] == "SIM-001"
    assert result["age"] == 45
    assert result["validation"]["valid"] is True
    assert result["validation"]["issues"] == []


def test_patient_agent_missing_symptoms():

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

    agent = PatientDataAgent()

    result = agent.process_patient(patient)

    assert result["validation"]["valid"] is False
    assert "No symptoms provided." in result["validation"]["issues"]