from pydantic import ValidationError

from app.models.schemas import PatientCase


class PatientDataAgent:
    """
    Validates and structures synthetic patient information
    for the Clinical Decision Support System.

    Educational simulation only.
    """

    def validate_patient(self, patient: PatientCase) -> dict:
        """Validate the basic patient information."""

        issues = []

        if not patient.patient_id:
            issues.append("Patient ID is missing.")

        if not patient.sex:
            issues.append("Sex is missing.")

        if not patient.symptoms:
            issues.append("No symptoms provided.")

        if patient.age < 0 or patient.age > 120:
            issues.append("Invalid patient age.")

        return {
            "patient_id": patient.patient_id,
            "valid": len(issues) == 0,
            "issues": issues,
        }

    def process_patient(self, patient: PatientCase) -> dict:
        """Create a structured patient summary."""

        validation = self.validate_patient(patient)

        return {
            "patient_id": patient.patient_id,
            "age": patient.age,
            "sex": patient.sex,
            "symptoms": patient.symptoms,
            "symptom_duration": patient.symptom_duration,
            "medical_history": patient.medical_history,
            "medications": patient.medications,
            "allergies": patient.allergies,
            "lab_results": patient.lab_results,
            "validation": validation,
        }
    def process_raw(self, data: dict) -> dict:
        """Build a PatientCase from raw input (e.g. JSON / UI form).

        Missing or malformed fields are reported as validation issues instead
        of raising, so the workflow can stop cleanly.
        """
        if not isinstance(data, dict):
            return {
                "patient": None,
                "validation": {"patient_id": None, "valid": False,
                               "issues": ["Patient input must be a JSON object."]},
            }
        try:
            patient = PatientCase(**data)
        except ValidationError as exc:
            issues = [
                f"{'.'.join(str(p) for p in err['loc']) or 'input'}: {err['msg']}"
                for err in exc.errors()
            ]
            return {
                "patient": None,
                "validation": {"patient_id": data.get("patient_id"), "valid": False,
                               "issues": issues},
            }
        return {"patient": patient, **self.process_patient(patient)}
