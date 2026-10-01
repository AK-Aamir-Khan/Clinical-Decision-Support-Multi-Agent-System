import pytest

from app.agents.safety_agent import REDACTION, SafetyReviewAgent, scan_text
from app.utils.validators import EDUCATIONAL_DISCLAIMER


def make(diff_text="Fever and cough are reported.", plan_text="Chest X-ray could be considered.",
         disclaimer=EDUCATIONAL_DISCLAIMER):
    differential = {"candidates": [{"condition": "Pneumonia", "supporting_evidence": [diff_text]}],
                    "reasoning_summary": "Possible considerations.", "disclaimer": disclaimer}
    plan = {"steps": [{"step": plan_text, "rationale": "Educational."}],
            "summary": "Steps to consider.", "disclaimer": disclaimer}
    return differential, plan


def test_clean_output_passes():
    result = SafetyReviewAgent().review(*make())
    assert result["passed"] is True
    assert result["risk_level"] == "low"
    assert result["requires_human_review"] is False


@pytest.mark.parametrize("text,kind", [
    ("The patient definitely has pneumonia.", "unsupported_certainty"),
    ("This is a confirmed diagnosis.", "unsupported_certainty"),
    ("Prescribe amoxicillin 500 mg three times daily.", "treatment_recommendation"),
    ("Start antibiotics immediately.", "treatment_recommendation"),
    ("Stop the medication.", "treatment_recommendation"),
    ("There is no need to see a doctor.", "unsafe_language"),
    ("You can self-medicate at home.", "unsafe_language"),
])
def test_violations_detected(text, kind):
    result = SafetyReviewAgent().review(*make(diff_text=text))
    assert result["passed"] is False
    assert kind in {i["issue_type"] for i in result["issues"]}
    assert result["feedback"]


def test_lab_units_are_not_mistaken_for_doses():
    assert scan_text("CRP is high (48 mg/L); glucose 130 mg/dL.") == []


def test_treatment_is_high_risk_and_flagged():
    result = SafetyReviewAgent().review(*make(plan_text="Give paracetamol 650 mg."))
    assert result["risk_level"] == "high"
    assert result["requires_human_review"] is True


def test_missing_disclaimer_detected():
    result = SafetyReviewAgent().review(*make(disclaimer=""))
    kinds = [i["issue_type"] for i in result["issues"]]
    assert kinds.count("missing_disclaimer") == 2
    assert result["risk_level"] == "moderate"


def test_red_flags_and_upstream_errors_raise_risk():
    result = SafetyReviewAgent().review(*make(), red_flags=["chest pain"], upstream_errors=["x"])
    assert result["passed"] is True  # the text itself is fine ...
    assert result["risk_level"] == "high"  # ... but the case needs human review
    assert result["requires_human_review"] is True
    assert any("chest pain" in n for n in result["notes"])


def test_sanitize_redacts_and_restores_disclaimer():
    diff, plan = make(diff_text="Definitely pneumonia.", plan_text="Prescribe antibiotics.", disclaimer="")
    clean_diff, clean_plan = SafetyReviewAgent.sanitize(diff, plan)
    assert REDACTION in clean_diff["candidates"][0]["supporting_evidence"][0]
    assert "Prescribe" not in clean_plan["steps"][0]["step"]
    assert clean_diff["disclaimer"] == clean_plan["disclaimer"] == EDUCATIONAL_DISCLAIMER
    assert diff["candidates"][0]["supporting_evidence"][0] == "Definitely pneumonia."  # original untouched
