"""Guardrails on the Differential Diagnosis Agent, tested with REAL replies
recorded from gemma3:4b (tests/data/gemma3_4b_recorded.json)."""

import json
from pathlib import Path

import pytest

from app.config import Settings
from app.workflow.graph import run_case
from tests.fakes import FakeStructuredLLM

ROOT = Path(__file__).resolve().parents[1]
RECORDED = json.loads((ROOT / "tests" / "data" / "gemma3_4b_recorded.json").read_text(encoding="utf-8"))
CASES = {c["patient_id"]: c for c in json.loads((ROOT / "data" / "synthetic_patients.json").read_text())}
OFFLINE = Settings(openai_api_key=None, use_llm=False)
PLAN = {"steps": [{"step": "An ECG could be considered", "rationale": "Educational.", "priority": "early"}],
        "summary": "Steps to consider."}


def run_with_recorded(patient_id):
    fake = FakeStructuredLLM({"DifferentialDiagnosisOutput": RECORDED[patient_id], "DiagnosticPlanOutput": PLAN})
    return run_case(CASES[patient_id], llm=fake, settings=OFFLINE)


def test_good_gemma_reply_passes_unchanged():
    diff = run_with_recorded("SIM-001")["differential_diagnosis"]
    assert diff["method"] == "llm"
    assert [c["condition"] for c in diff["candidates"]] == [
        "Community-acquired pneumonia", "Influenza-like illness", "Urinary tract infection"]
    assert diff["guardrail_notes"] == []


def test_gemma_red_flag_case_is_corrected():
    state = run_with_recorded("SIM-005")
    diff = state["differential_diagnosis"]
    levels = {c["condition"]: c["consideration_level"] for c in diff["candidates"]}

    # the warning-symptom explanation is promoted to the top
    assert diff["candidates"][0]["condition"] == "Acute coronary syndrome"
    assert levels["Acute coronary syndrome"] == "higher"
    # already-known diabetes is context, not the main explanation
    assert levels["Type 2 diabetes mellitus"] == "lower"
    # ungrounded candidates are removed
    assert "Hypertension" not in levels  # no knowledge reference
    assert "Dengue fever" not in levels  # nothing in the case matches the dengue document
    assert len(diff["guardrail_notes"]) == 4
    assert state["safety_review"]["risk_level"] == "high"  # red flags still escalate


def test_rule_based_output_also_gets_known_history_cap():
    diff = run_case(CASES["SIM-005"], settings=OFFLINE)["differential_diagnosis"]
    assert diff["candidates"][0]["condition"].startswith("Acute coronary syndrome")
    diabetes = next(c for c in diff["candidates"] if "diabetes" in c["condition"].lower())
    assert diabetes["consideration_level"] == "lower"


@pytest.mark.parametrize("patient_id", ["SIM-001", "SIM-002", "SIM-003", "SIM-004", "SIM-006", "SIM-007"])
def test_guardrails_do_not_change_rule_based_top_candidate(patient_id):
    diff = run_case(CASES[patient_id], settings=OFFLINE)["differential_diagnosis"]
    assert diff["candidates"][0]["consideration_level"] == "higher"


def test_gemma_hallucinated_evidence_is_removed_after_prompt_fix():
    """With the tightened prompt Gemma ranks ACS first, but still invents a
    'location (India)' risk factor for dengue - the grounding check removes it."""
    fake = FakeStructuredLLM({"DifferentialDiagnosisOutput": RECORDED["SIM-005-tightened-prompt"],
                              "DiagnosticPlanOutput": PLAN})
    diff = run_case(CASES["SIM-005"], llm=fake, settings=OFFLINE)["differential_diagnosis"]
    levels = {c["condition"]: c["consideration_level"] for c in diff["candidates"]}

    assert diff["candidates"][0]["condition"] == "Acute coronary syndrome"
    assert "Dengue fever" not in levels
    assert levels["Type 2 diabetes mellitus"] == "lower"


def test_gemma_plan_necessity_wording_is_caught_and_red_flag_step_added():
    """Real gemma3:4b planning reply: sensible steps, but 'is warranted' / 'should be
    measured' / 'strongly support the diagnosis' break the could-be-considered rule,
    and the 'prompt clinician review' label is put on an ECG step."""
    fake = FakeStructuredLLM({"DifferentialDiagnosisOutput": RECORDED["SIM-005-tightened-prompt"],
                              "DiagnosticPlanOutput": RECORDED["PLAN-SIM-005"]})
    state = run_case(CASES["SIM-005"], llm=fake, settings=OFFLINE)
    plan, safety, report = state["diagnostic_plan"], state["safety_review"], state["final_report"]

    assert plan["steps"][0]["step"].startswith("Prompt in-person clinician review")
    excerpts = {i["excerpt"].lower() for i in safety["issues"]}
    assert {"is warranted", "should be measured", "strongly support the diagnosis"} <= excerpts
    assert state["revision_count"] == 1  # feedback was sent back once
    planning_prompts = [c for c in fake.calls if c[0] == "DiagnosticPlanOutput"]
    assert "should be measured" in planning_prompts[1][1][1][1]  # planner received the feedback
    assert "should be measured" not in report["markdown"]  # still present after revision -> redacted
