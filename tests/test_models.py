from app.models.schemas import PatientCase


def test_patient_case():
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

    assert patient.patient_id == "SIM-001"
    assert patient.age == 45
    assert "fever" in patient.symptoms