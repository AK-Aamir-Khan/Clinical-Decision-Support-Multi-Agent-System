from app.agents.diagnostic_planning_agent import DiagnosticPlanningAgent
from app.models.schemas import DiagnosticPlanOutput
from app.utils.validators import EDUCATIONAL_DISCLAIMER
from tests.fakes import FakeStructuredLLM

SYMPTOMS = {"key_features": ["fever", "cough"], "red_flags": []}
HISTORY = {"conditions": []}
LABS = {"results": [], "abnormal_findings": [], "invalid_values": []}
DIFFERENTIAL = {"candidates": [
    {"condition": "Community-acquired pneumonia", "consideration_level": "higher",
     "supporting_evidence": ["fever"], "knowledge_refs": ["KB-001"]},
    {"condition": "Interpreting inflammatory markers", "consideration_level": "lower",
     "supporting_evidence": [], "knowledge_refs": ["KB-012"]},
]}


def test_rule_based_plan_structure():
    result = DiagnosticPlanningAgent().plan(SYMPTOMS, HISTORY, LABS, DIFFERENTIAL)

    DiagnosticPlanOutput.model_validate(result)
    assert result["method"] == "rule-based"
    assert result["disclaimer"] == EDUCATIONAL_DISCLAIMER
    assert any("Chest X-ray" in s["step"] for s in result["steps"])
    first = result["steps"][0]
    assert first["rationale"] and first["related_candidates"] == ["Community-acquired pneumonia"]
    assert first["priority"] == "early"
    assert len(result["steps"]) <= 8


def test_red_flags_put_clinician_review_first():
    symptoms = {"key_features": ["chest pain"], "red_flags": ["chest pain"]}
    result = DiagnosticPlanningAgent().plan(symptoms, HISTORY, LABS, DIFFERENTIAL)
    assert result["steps"][0]["priority"] == "prompt clinician review"
    assert "chest pain" in result["steps"][0]["rationale"]


def test_invalid_labs_and_empty_candidates():
    labs = {**LABS, "invalid_values": ["WBC"]}
    result = DiagnosticPlanningAgent().plan(SYMPTOMS, HISTORY, labs, {"candidates": []})
    assert len(result["steps"]) == 1
    assert "WBC" in result["steps"][0]["step"]


def test_llm_plan_used_and_red_flag_enforced():
    fake = FakeStructuredLLM({"DiagnosticPlanOutput": {
        "steps": [{"step": "ECG could be considered", "rationale": "Assess cardiac rhythm.",
                   "related_candidates": ["Acute coronary syndrome"], "priority": "early"}],
        "summary": "LLM plan.",
    }})
    symptoms = {"key_features": ["chest pain"], "red_flags": ["chest pain"]}
    result = DiagnosticPlanningAgent(llm=fake).plan(symptoms, HISTORY, LABS, DIFFERENTIAL)

    assert result["method"] == "llm"
    assert result["steps"][0]["priority"] == "prompt clinician review"
    assert result["steps"][1]["step"] == "ECG could be considered"


def test_llm_failure_falls_back():
    fake = FakeStructuredLLM({"DiagnosticPlanOutput": TimeoutError("timed out")})
    result = DiagnosticPlanningAgent(llm=fake).plan(SYMPTOMS, HISTORY, LABS, DIFFERENTIAL)
    assert result["method"] == "rule-based"
    assert "TimeoutError" in result["llm_error"]
