"""Workflow-level tests: routing, shared state and error handling."""

from app.workflow.graph import build_graph, run_case

VALID_INPUT = {
    "patient_id": "SIM-001", "age": 45, "sex": "Male",
    "symptoms": ["fever", "cough"], "symptom_duration": "5 days",
    "lab_results": {"WBC": 13500},
}


def test_graph_runs_parallel_analyses():
    state = run_case(VALID_INPUT)
    agents = [t["agent"] for t in state["agent_trace"]]

    assert state["validation"]["valid"] is True
    assert agents[0] == "Patient Data Agent"
    assert set(agents[1:4]) == {"Symptom Analysis Agent", "Medical History Agent", "Laboratory Analysis Agent"}
    assert agents[4:] == ["RAG Retriever", "Differential Diagnosis Agent", "Diagnostic Planning Agent",
                          "Safety Review Agent", "Report Generator"]


def test_graph_stops_on_invalid_input():
    state = run_case({"patient_id": "SIM-BAD", "age": 200})

    assert state["validation"]["valid"] is False
    assert any("age" in issue for issue in state["validation"]["issues"])
    assert "symptom_analysis" not in state
    assert state["final_report"]["status"] == "stopped_invalid_input"
    assert [t["agent"] for t in state["agent_trace"]] == ["Patient Data Agent", "Report Generator"]


def test_graph_compiles_with_expected_nodes():
    nodes = set(build_graph().get_graph().nodes)
    assert {"patient_data", "knowledge_retrieval", "safety_review", "final_report"} <= nodes


def test_run_case_handles_unexpected_errors():
    class BrokenGraph:
        def invoke(self, _):
            raise RuntimeError("boom with sk-secretsecretsecret")

    state = run_case(VALID_INPUT, graph=BrokenGraph())
    assert state["final_report"]["status"] == "workflow_error"
    assert "sk-secretsecretsecret" not in state["errors"][0]
