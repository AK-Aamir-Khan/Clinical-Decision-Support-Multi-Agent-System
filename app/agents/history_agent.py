"""Medical History Agent.

Reviews past conditions, medications and allergies of a synthetic patient
and highlights history-related factors that may matter for the educational
case. It never recommends changing, starting or stopping any medication.
"""

from typing import Dict, List, Tuple

from app.models.schemas import PatientCase
from app.utils.validators import normalize_term

# keyword -> (category, general educational relevance note)
CONDITION_KNOWLEDGE: Dict[str, Tuple[str, str]] = {
    "hypertension": ("cardiovascular", "Blood pressure history can be relevant when interpreting cardiovascular and kidney-related findings."),
    "heart failure": ("cardiovascular", "Cardiac history can be relevant when breathlessness or fluid-related symptoms are present."),
    "coronary artery disease": ("cardiovascular", "Known coronary disease makes cardiac causes important to keep in mind for chest symptoms."),
    "diabetes": ("metabolic/endocrine", "Relevant when interpreting glucose/HbA1c results; diabetes is commonly listed as a factor that can increase infection risk."),
    "hypothyroidism": ("endocrine", "Relevant when interpreting thyroid function tests (e.g. TSH)."),
    "hyperthyroidism": ("endocrine", "Relevant when interpreting thyroid function tests (e.g. TSH)."),
    "asthma": ("respiratory", "Relevant when respiratory symptoms such as wheeze, cough or breathlessness are present."),
    "copd": ("respiratory", "Chronic lung disease is relevant when respiratory symptoms are present."),
    "chronic kidney disease": ("renal", "Relevant when interpreting creatinine and electrolyte results."),
    "ckd": ("renal", "Relevant when interpreting creatinine and electrolyte results."),
    "anemia": ("hematologic", "Relevant when interpreting hemoglobin and related indices."),
    "anaemia": ("hematologic", "Relevant when interpreting hemoglobin and related indices."),
    "migraine": ("neurological", "A prior headache pattern is relevant context when headache is reported."),
    "hiv": ("immune status", "Reduced immune function can widen the range of infections considered."),
    "chemotherapy": ("immune status", "Reduced immune function can widen the range of infections considered."),
    "smok": ("lifestyle risk factor", "Smoking is a recognised risk factor for respiratory and cardiovascular conditions."),
    "pregnan": ("special population", "Pregnancy changes reference ranges and the considerations clinicians apply."),
    "menorrhagia": ("gynecological", "Heavy menstrual bleeding is a recognised cause of iron loss."),
    "heavy menstrual": ("gynecological", "Heavy menstrual bleeding is a recognised cause of iron loss."),
}

# keyword -> general medication class (context only, never advice)
MEDICATION_CLASSES: Dict[str, str] = {
    "metformin": "glucose-lowering (biguanide)",
    "insulin": "glucose-lowering (insulin)",
    "amlodipine": "antihypertensive (calcium channel blocker)",
    "lisinopril": "antihypertensive (ACE inhibitor)",
    "ramipril": "antihypertensive (ACE inhibitor)",
    "enalapril": "antihypertensive (ACE inhibitor)",
    "losartan": "antihypertensive (angiotensin receptor blocker)",
    "telmisartan": "antihypertensive (angiotensin receptor blocker)",
    "antihypertensive": "antihypertensive (class not specified)",
    "levothyroxine": "thyroid hormone replacement",
    "atorvastatin": "lipid-lowering (statin)",
    "rosuvastatin": "lipid-lowering (statin)",
    "aspirin": "antiplatelet",
    "warfarin": "anticoagulant",
    "salbutamol": "bronchodilator (inhaled)",
    "albuterol": "bronchodilator (inhaled)",
    "inhaler": "inhaled respiratory medication",
    "prednisolone": "corticosteroid",
    "prednisone": "corticosteroid",
    "oral contraceptive": "hormonal contraceptive",
}

# medication classes that are worth flagging as context for the case
MEDICATION_FLAGS: Dict[str, str] = {
    "corticosteroid": "Corticosteroid use can affect immune response and blood glucose.",
    "anticoagulant": "Anticoagulant use is relevant context for any bleeding-related findings.",
    "glucose-lowering": "Glucose-lowering medication is relevant context when interpreting glucose results.",
}

HISTORY_NOTE = (
    "History review for an educational simulation. Medication information is "
    "recorded for context only; no medication changes are suggested."
)


class MedicalHistoryAgent:
    """Analyzes medical history, medications and allergies."""

    def _classify_condition(self, condition: str) -> Dict[str, str]:
        term = normalize_term(condition)
        for keyword, (category, relevance) in CONDITION_KNOWLEDGE.items():
            if keyword in term:
                return {"name": condition, "category": category, "relevance": relevance}
        return {
            "name": condition,
            "category": "unclassified",
            "relevance": "Not in the project's small history catalogue; recorded as reported.",
        }

    def _classify_medication(self, medication: str) -> Dict[str, str]:
        term = normalize_term(medication)
        for keyword, drug_class in MEDICATION_CLASSES.items():
            if keyword in term:
                return {"name": medication, "drug_class": drug_class}
        return {"name": medication, "drug_class": "unclassified"}

    @staticmethod
    def _age_factors(age: int) -> List[str]:
        if age >= 65:
            return ["Older adult age group (65+): presentations can be atypical."]
        if age < 5:
            return ["Young child age group (<5): age-specific reference ranges apply."]
        return []

    def analyze(self, patient: PatientCase) -> dict:
        conditions = [self._classify_condition(c) for c in patient.medical_history if c.strip()]
        medications = [self._classify_medication(m) for m in patient.medications if m.strip()]
        allergies = [a.strip() for a in patient.allergies if a.strip()]

        risk_factors = self._age_factors(patient.age)
        risk_factors += [
            f"{c['name']} ({c['category']})"
            for c in conditions
            if c["category"] != "unclassified"
        ]

        history_flags: List[str] = []
        for med in medications:
            for class_key, flag in MEDICATION_FLAGS.items():
                if class_key in med["drug_class"] and flag not in history_flags:
                    history_flags.append(flag)
        if allergies:
            history_flags.append(
                "Documented allergies: " + ", ".join(allergies)
                + ". Highlighted for clinician awareness."
            )

        summary = (
            f"{len(conditions)} past condition(s), {len(medications)} medication(s) "
            f"and {len(allergies)} allergy record(s) reviewed."
        )
        if not (conditions or medications or allergies):
            summary = "No past medical history, medications or allergies were reported."

        return {
            "patient_id": patient.patient_id,
            "conditions": conditions,
            "medications": medications,
            "allergies": allergies,
            "risk_factors": risk_factors,
            "history_flags": history_flags,
            "summary": summary,
            "note": HISTORY_NOTE,
        }
