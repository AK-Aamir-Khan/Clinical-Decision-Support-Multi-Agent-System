"""End-to-end LangGraph workflow tests on every synthetic case (no API calls)."""

import json
from pathlib import Path

import pytest

from app.agents.safety_agent import REDACTION
from app.config import Settings
from app.utils.validators import EDUCATIONAL_DISCLAIMER
from app.workflow import nodes
from app.workflow.graph import run_case
from tests.fakes import FakeStructuredLLM

CASES = json.loads((Path(__file__).resolve().parents[1] / "data" / "synthetic_patients.json").read_text())
VALID_CASES = [c for c in CASES if c["patient_id"] != "SIM-008"]
OFFLINE = Settings(openai_api_key=None, use_llm=False)


def case(patient_id):
    return next(c for c in CASES if c["patient_id"] == patient_id)


def test_synthetic_data_is_clearly_synthetic():
    assert len(VALID_CASES) >= 5
    assert all(c["patient_id"].startswith("SIM-") for c in CASES)
    for c in CASES:  # no identifying fields
        assert not {"name", "phone", "address", "email"} & set(c)


@pytest.mark.parametrize("patient", VALID_CASES, ids=lambda c: c["patient_id"])
def test_every_valid_case_completes_offline(patient):
    state = run_case(patient, settings=OFFLINE)
    report = state["final_report"]

    assert report["status"] == "completed"
    assert report["disclaimer"] == EDUCATIONAL_DISCLAIMER
    assert state["retrieved_knowledge"], "retrieval should find context for each demo case"
    assert report["differential_diagnosis"]
    assert report["diagnostic_plan"]
    assert state["safety_review"]["passed"] is True  # rule-based text is safe
    assert all(t["status"] == "ok" for t in state["agent_trace"])
    assert EDUCATIONAL_DISCLAIMER in report["markdown"]


@pytest.mark.parametrize("patient_id,expected_top", [
    ("SIM-001", "KB-001"), ("SIM-002", "KB-006"), ("SIM-003", "KB-007"),
    ("SIM-004", "KB-004"), ("SIM-006", "KB-008"), ("SIM-007", "KB-005"),
])
def test_expected_top_candidate(patient_id, expected_top):
    report = run_case(case(patient_id), settings=OFFLINE)["final_report"]
    assert report["differential_diagnosis"][0]["knowledge_refs"] == [expected_top]


def test_red_flag_case_is_high_risk():
    state = run_case(case("SIM-005"), settings=OFFLINE)
    assert state["safety_review"]["risk_level"] == "high"
    assert state["final_report"]["safety"]["requires_human_review"] is True
    assert state["diagnostic_plan"]["steps"][0]["priority"] == "prompt clinician review"


def test_invalid_lab_value_is_reported():
    state = run_case(case("SIM-007"), settings=OFFLINE)
    assert state["lab_analysis"]["invalid_values"] == ["CRP"]
    assert any("CRP" in s["step"] for s in state["diagnostic_plan"]["steps"])


def test_invalid_patient_stops_workflow():
    report = run_case(case("SIM-008"), settings=OFFLINE)["final_report"]
    assert report["status"] == "stopped_invalid_input"
    assert len(report["validation_issues"]) == 2


GOOD_DIFF = {"candidates": [{"condition": "Lower respiratory tract infection", "consideration_level": "higher",
                             "supporting_evidence": ["fever", "cough", "WBC high"], "knowledge_refs": ["KB-001"]}],
             "reasoning_summary": "A possibility to consider."}
BAD_DIFF = {"candidates": [{"condition": "Pneumonia", "consideration_level": "higher",
                            "supporting_evidence": ["The patient definitely has pneumonia"],
                            "knowledge_refs": ["KB-001"]}],
            "reasoning_summary": "Confirmed diagnosis."}
GOOD_PLAN = {"steps": [{"step": "Chest X-ray could be considered", "rationale": "Looks for consolidation.",
                        "related_candidates": ["Lower respiratory tract infection"], "priority": "early"}],
             "summary": "Steps to consider."}


def test_llm_workflow_with_mocked_llm():
    fake = FakeStructuredLLM({"DifferentialDiagnosisOutput": GOOD_DIFF, "DiagnosticPlanOutput": GOOD_PLAN})
    state = run_case(case("SIM-001"), llm=fake, settings=OFFLINE)

    assert state["llm_mode"] == "llm"
    assert state["differential_diagnosis"]["method"] == "llm"
    assert state["diagnostic_plan"]["method"] == "llm"
    assert state["safety_review"]["passed"] is True
    assert state["revision_count"] == 0


def test_safety_feedback_loop_revises_llm_output():
    fake = FakeStructuredLLM({"DifferentialDiagnosisOutput": [BAD_DIFF, GOOD_DIFF],
                              "DiagnosticPlanOutput": GOOD_PLAN})
    state = run_case(case("SIM-001"), llm=fake, settings=OFFLINE)
    agents = [t["agent"] for t in state["agent_trace"]]

    assert state["revision_count"] == 1
    assert agents.count("Differential Diagnosis Agent") == 2
    assert agents.count("Safety Review Agent") == 2
    # the second diagnosis call received the safety feedback in its prompt
    second_prompt = [c for c in fake.calls if c[0] == "DifferentialDiagnosisOutput"][1][1][1][1]
    assert "unsupported_certainty" in second_prompt
    assert state["safety_review"]["passed"] is True


def test_persistent_violation_is_redacted_in_report():
    fake = FakeStructuredLLM({"DifferentialDiagnosisOutput": BAD_DIFF, "DiagnosticPlanOutput": GOOD_PLAN})
    state = run_case(case("SIM-001"), llm=fake, settings=OFFLINE)
    report = state["final_report"]

    assert state["revision_count"] == 1  # only one revision allowed
    assert state["safety_review"]["passed"] is False
    assert "definitely" not in report["markdown"].lower()
    assert REDACTION in report["markdown"]


def test_llm_api_failure_falls_back_and_completes():
    fake = FakeStructuredLLM({"DifferentialDiagnosisOutput": ConnectionError("API down"),
                              "DiagnosticPlanOutput": ConnectionError("API down")})
    state = run_case(case("SIM-001"), llm=fake, settings=OFFLINE)

    assert state["final_report"]["status"] == "completed"
    assert state["differential_diagnosis"]["method"] == "rule-based"
    assert any("fallback" in e for e in state["errors"])
    assert state["safety_review"]["risk_level"] == "moderate"  # upstream errors noted


def test_agent_exception_is_contained(monkeypatch):
    def explode(self, patient):
        raise ValueError("lab service crashed")

    monkeypatch.setattr(nodes.LaboratoryAnalysisAgent, "analyze", explode)
    state = run_case(case("SIM-001"), settings=OFFLINE)

    assert state["final_report"]["status"] == "completed"
    lab_trace = next(t for t in state["agent_trace"] if t["agent"] == "Laboratory Analysis Agent")
    assert lab_trace["status"] == "error"
    assert any("lab service crashed" in e for e in state["errors"])
