from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from app.utils.validators import EDUCATIONAL_DISCLAIMER


class PatientCase(BaseModel):
    patient_id: str
    age: int = Field(ge=0, le=120)
    sex: str
    symptoms: List[str]
    symptom_duration: Optional[str] = None
    medical_history: List[str] = []
    medications: List[str] = []
    allergies: List[str] = []
    lab_results: Dict[str, float] = {}


# ---------------------------------------------------------------------------
# Structured outputs of the reasoning agents (also used as LLM output schemas)
# ---------------------------------------------------------------------------


class DiagnosisCandidate(BaseModel):
    """One condition/category to *consider* - never a confirmed diagnosis."""

    condition: str
    category: str = "general"
    consideration_level: Literal["higher", "moderate", "lower"] = Field(
        description="Relative level of consideration in this simulated case, not a probability."
    )
    supporting_evidence: List[str] = Field(default_factory=list)
    missing_or_conflicting_evidence: List[str] = Field(default_factory=list)
    knowledge_refs: List[str] = Field(default_factory=list, description="IDs of retrieved knowledge documents used.")


class DifferentialDiagnosisOutput(BaseModel):
    candidates: List[DiagnosisCandidate] = Field(default_factory=list, max_length=5)
    reasoning_summary: str
    disclaimer: str = EDUCATIONAL_DISCLAIMER


class DiagnosticStep(BaseModel):
    step: str = Field(description="A diagnostic test or assessment that could be considered.")
    rationale: str
    related_candidates: List[str] = Field(default_factory=list)
    priority: Literal["prompt clinician review", "early", "routine"] = "routine"


class DiagnosticPlanOutput(BaseModel):
    steps: List[DiagnosticStep] = Field(default_factory=list, max_length=8)
    summary: str
    disclaimer: str = EDUCATIONAL_DISCLAIMER


class SafetyIssue(BaseModel):
    issue_type: Literal["unsupported_certainty", "treatment_recommendation", "unsafe_language", "missing_disclaimer"]
    excerpt: str
    location: str


class SafetyReviewOutput(BaseModel):
    passed: bool
    risk_level: Literal["low", "moderate", "high"]
    requires_human_review: bool
    issues: List[SafetyIssue] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
