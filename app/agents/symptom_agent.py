import re
from typing import List, Optional

from app.models.schemas import PatientCase

# General warning features that, in real care settings, usually prompt
# timely clinician evaluation. Used only to raise the educational risk level.
RED_FLAG_TERMS = [
    "chest pain", "shortness of breath", "difficulty breathing", "breathlessness at rest",
    "confusion", "fainting", "syncope", "seizure", "slurred speech",
    "weakness on one side", "coughing blood", "vomiting blood", "blood in stool",
    "severe headache", "stiff neck",
]

BODY_SYSTEMS = {
    "fever": "constitutional", "fatigue": "constitutional", "weight loss": "constitutional",
    "chills": "constitutional", "night sweats": "constitutional", "weight gain": "constitutional",
    "cough": "respiratory", "shortness of breath": "respiratory", "breathlessness": "respiratory",
    "wheez": "respiratory", "sore throat": "respiratory", "runny nose": "respiratory",
    "chest pain": "cardiovascular", "palpitation": "cardiovascular",
    "headache": "neurological", "dizziness": "neurological", "confusion": "neurological",
    "abdominal pain": "gastrointestinal", "nausea": "gastrointestinal",
    "vomiting": "gastrointestinal", "diarrh": "gastrointestinal", "constipation": "gastrointestinal",
    "burning urination": "genitourinary", "dysuria": "genitourinary",
    "frequent urination": "genitourinary", "urinary": "genitourinary",
    "thirst": "metabolic/endocrine", "cold intolerance": "metabolic/endocrine",
    "rash": "dermatological", "joint pain": "musculoskeletal", "body ache": "musculoskeletal",
    "pallor": "hematologic", "bruising": "hematologic", "bleeding gums": "hematologic",
}

_UNIT_DAYS = {"minute": 0, "hour": 0, "hr": 0, "day": 1, "week": 7, "month": 30, "year": 365}


def parse_duration_days(duration: Optional[str]) -> Optional[int]:
    """Convert strings like '5 days' or '3 weeks' to an approximate day count."""
    if not duration:
        return None
    match = re.search(r"(\d+)\s*(minute|hour|hr|day|week|month|year)", duration.lower())
    if not match:
        return None
    return int(match.group(1)) * _UNIT_DAYS[match.group(2)]


def duration_category(days: Optional[int]) -> str:
    """Descriptive time-course label (not a clinical classification)."""
    if days is None:
        return "not specified"
    if days == 0:
        return "very recent (less than a day)"
    if days <= 14:
        return "short (up to 2 weeks)"
    if days <= 90:
        return "intermediate (2 weeks to 3 months)"
    return "long-standing (over 3 months)"


class SymptomAnalysisAgent:
    """
    Analyzes patient symptoms and produces a structured
    symptom assessment for educational simulation.

    This agent does not provide a medical diagnosis.
    """

    def analyze(self, patient: PatientCase) -> dict:
        symptoms = patient.symptoms

        if not symptoms:
            return {
                "patient_id": patient.patient_id,
                "symptoms": [],
                "symptom_count": 0,
                "duration": patient.symptom_duration,
                "analysis": "No symptoms were provided.",
                "key_features": [],
                "duration_days": None,
                "duration_category": "not specified",
                "body_systems": [],
                "red_flags": [],
            }

        key_features = []

        for symptom in symptoms:
            key_features.append(symptom.strip().lower())

        analysis = (
            f"The patient reports {len(symptoms)} symptom(s): "
            f"{', '.join(key_features)}."
        )

        if patient.symptom_duration:
            analysis += (
                f" The reported symptom duration is "
                f"{patient.symptom_duration}."
            )

        days = parse_duration_days(patient.symptom_duration)

        return {
            "patient_id": patient.patient_id,
            "symptoms": symptoms,
            "symptom_count": len(symptoms),
            "duration": patient.symptom_duration,
            "analysis": analysis,
            "key_features": key_features,
            "duration_days": days,
            "duration_category": duration_category(days),
            "body_systems": self._body_systems(key_features),
            "red_flags": self._red_flags(key_features),
        }

    @staticmethod
    def _body_systems(features: List[str]) -> List[str]:
        systems = []
        for feature in features:
            for keyword, system in BODY_SYSTEMS.items():
                if keyword in feature and system not in systems:
                    systems.append(system)
        return systems

    @staticmethod
    def _red_flags(features: List[str]) -> List[str]:
        return [f for f in features if any(term in f for term in RED_FLAG_TERMS)]