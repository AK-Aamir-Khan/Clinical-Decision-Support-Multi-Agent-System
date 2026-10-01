import pytest

from app.agents.diagnosis_agent import DifferentialDiagnosisAgent
from app.agents.history_agent import MedicalHistoryAgent
from app.agents.lab_agent import LaboratoryAnalysisAgent
from app.agents.symptom_agent import SymptomAnalysisAgent
from app.models.schemas import DifferentialDiagnosisOutput
from app.rag.retriever import Retriever, build_retrieval_query
from app.utils.validators import EDUCATIONAL_DISCLAIMER
from tests.fakes import FakeStructuredLLM


@pytest.fixture
def upstream(respiratory_case):
    symptoms = SymptomAnalysisAgent().analyze(respiratory_case)
    history = MedicalHistoryAgent().analyze(respiratory_case)
    labs = LaboratoryAnalysisAgent().analyze(respiratory_case)
    q = build_retrieval_query(symptoms, history, labs)
    retrieved = Retriever().retrieve(q["query"], lab_signals=q["lab_signals"])
    return symptoms, history, labs, retrieved


def test_rule_based_output_structure(upstream):
    result = DifferentialDiagnosisAgent().diagnose(*upstream)

    DifferentialDiagnosisOutput.model_validate(result)  # structure is valid
    assert result["method"] == "rule-based"
    assert result["disclaimer"] == EDUCATIONAL_DISCLAIMER
    assert 1 <= len(result["candidates"]) <= 5
    top = result["candidates"][0]
    assert top["condition"] == "Community-acquired pneumonia"
    assert top["consideration_level"] in ("higher", "moderate", "lower")
    assert any("WBC high" in e for e in top["supporting_evidence"])


def test_empty_retrieval_gives_no_candidates(upstream):
    symptoms, history, labs, _ = upstream
    result = DifferentialDiagnosisAgent().diagnose(symptoms, history, labs, [])
    assert result["candidates"] == []
    assert "Insufficient" in result["reasoning_summary"]


def test_llm_output_is_used_and_refs_filtered(upstream):
    fake = FakeStructuredLLM({"DifferentialDiagnosisOutput": {
        "candidates": [{
            "condition": "Lower respiratory tract infection",
            "consideration_level": "higher",
            "supporting_evidence": ["fever", "cough", "raised WBC"],
            "knowledge_refs": ["KB-001", "KB-999"],
        }],
        "reasoning_summary": "Educational reasoning.",
        "disclaimer": "wrong disclaimer",
    }})
    result = DifferentialDiagnosisAgent(llm=fake).diagnose(*upstream, safety_feedback=["Remove certainty"])

    assert result["method"] == "llm"
    assert result["candidates"][0]["knowledge_refs"] == ["KB-001"]
    assert result["disclaimer"] == EDUCATIONAL_DISCLAIMER
    # the prompt only carries structured upstream context + feedback
    human_prompt = fake.calls[0][1][1][1]
    assert "Remove certainty" in human_prompt and "KB-001" in human_prompt


@pytest.mark.parametrize("bad_response", [
    RuntimeError("API connection failed for key sk-abcdefghijklmnop"),
    {"candidates": "not-a-list"},  # malformed output -> validation error
])
def test_llm_failure_falls_back_to_rules(upstream, bad_response):
    fake = FakeStructuredLLM({"DifferentialDiagnosisOutput": bad_response})
    result = DifferentialDiagnosisAgent(llm=fake).diagnose(*upstream)

    assert result["method"] == "rule-based"
    assert result["llm_error"]
    assert "sk-abcdefghijklmnop" not in result["llm_error"]
    assert result["candidates"]
