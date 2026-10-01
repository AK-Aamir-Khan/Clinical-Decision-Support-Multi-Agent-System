"""Shared pytest fixtures (synthetic data only)."""

import pytest

from app.models.schemas import PatientCase


@pytest.fixture
def respiratory_case() -> PatientCase:
    return PatientCase(
        patient_id="SIM-001",
        age=45,
        sex="Male",
        symptoms=["fever", "cough", "fatigue"],
        symptom_duration="5 days",
        medical_history=["hypertension"],
        medications=["antihypertensive medication"],
        allergies=[],
        lab_results={"WBC": 13500, "CRP": 48},
    )


@pytest.fixture
def empty_history_case() -> PatientCase:
    return PatientCase(
        patient_id="SIM-X",
        age=30,
        sex="Female",
        symptoms=["headache"],
    )
