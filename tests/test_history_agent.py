from app.agents.history_agent import MedicalHistoryAgent
from app.models.schemas import PatientCase


def test_history_classifies_condition_and_medication(respiratory_case):
    result = MedicalHistoryAgent().analyze(respiratory_case)

    assert result["patient_id"] == "SIM-001"
    assert result["conditions"][0]["category"] == "cardiovascular"
    assert "antihypertensive" in result["medications"][0]["drug_class"]
    assert any("hypertension" in r for r in result["risk_factors"])


def test_history_empty(empty_history_case):
    result = MedicalHistoryAgent().analyze(empty_history_case)

    assert result["conditions"] == []
    assert result["medications"] == []
    assert "No past medical history" in result["summary"]


def test_history_flags_allergy_steroid_and_age():
    patient = PatientCase(
        patient_id="SIM-H",
        age=70,
        sex="Female",
        symptoms=["cough"],
        medical_history=["Type 2 diabetes", "rare condition xyz"],
        medications=["Prednisolone"],
        allergies=["penicillin"],
    )
    result = MedicalHistoryAgent().analyze(patient)

    categories = {c["name"]: c["category"] for c in result["conditions"]}
    assert categories["Type 2 diabetes"] == "metabolic/endocrine"
    assert categories["rare condition xyz"] == "unclassified"
    assert any("Older adult" in r for r in result["risk_factors"])
    assert any("Corticosteroid" in f for f in result["history_flags"])
    assert any("penicillin" in f for f in result["history_flags"])


def test_history_never_recommends_medication_change(respiratory_case):
    result = MedicalHistoryAgent().analyze(respiratory_case)
    text = str(result).lower()
    for phrase in ("stop taking", "increase dose", "start taking", "prescribe"):
        assert phrase not in text
