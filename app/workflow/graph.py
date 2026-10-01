"""LangGraph orchestration of the MedAgent-CDSS agents.

    START
      -> patient_data                          (validate + structure input)
      -> [symptom_analysis | history_analysis | lab_analysis]   (parallel)
      -> knowledge_retrieval                   (RAG layer, waits for all 3)
      -> differential_diagnosis
      -> diagnostic_planning
      -> safety_review --(issues in LLM output, max N times)--> differential_diagnosis
      -> final_report
    END

Invalid patient input skips straight to final_report.
Agents communicate only through the shared ClinicalState.
"""

from typing import Optional

from langgraph.graph import END, START, StateGraph

from app.config import Settings, get_settings
from app.utils.validators import EDUCATIONAL_DISCLAIMER, sanitize_error
from app.workflow.nodes import build_analysis_nodes, build_reasoning_nodes
from app.workflow.state import ClinicalState

ANALYSIS_NODES = ["symptom_analysis", "history_analysis", "lab_analysis"]


def route_after_validation(state: ClinicalState):
    """Fan out to the three analysis agents only if the input is valid."""
    if state.get("validation", {}).get("valid") and state.get("patient"):
        return ANALYSIS_NODES
    return "final_report"


def route_after_safety(state: ClinicalState) -> str:
    """Safety agent either sends feedback back to the diagnosis agent or finalises."""
    if state.get("safety_review", {}).get("action") == "revise":
        return "differential_diagnosis"
    return "final_report"


def build_graph(llm=None, settings: Optional[Settings] = None):
    settings = settings or get_settings()
    graph = StateGraph(ClinicalState)
    for name, fn in {**build_analysis_nodes(), **build_reasoning_nodes(llm, settings)}.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "patient_data")
    graph.add_conditional_edges("patient_data", route_after_validation, ANALYSIS_NODES + ["final_report"])
    graph.add_edge(ANALYSIS_NODES, "knowledge_retrieval")  # fan-in: waits for all three
    graph.add_edge("knowledge_retrieval", "differential_diagnosis")
    graph.add_edge("differential_diagnosis", "diagnostic_planning")
    graph.add_edge("diagnostic_planning", "safety_review")
    graph.add_conditional_edges("safety_review", route_after_safety, ["differential_diagnosis", "final_report"])
    graph.add_edge("final_report", END)
    return graph.compile()


def run_case(patient_input: dict, llm=None, settings: Optional[Settings] = None, graph=None) -> dict:
    """Run one synthetic case through the workflow and return the final state.

    Unexpected workflow-level errors are caught and returned as a safe report.
    """
    initial = {
        "patient_input": patient_input,
        "revision_count": 0,
        "safety_feedback": [],
        "llm_mode": "llm" if llm is not None else "rule-based",
        "llm_model": getattr(llm, "model_name", None),
        "agent_trace": [],
        "errors": [],
    }
    try:
        graph = graph or build_graph(llm=llm, settings=settings)
        return graph.invoke(initial)
    except Exception as exc:  # noqa: BLE001
        error = f"Workflow error: {sanitize_error(exc)}"
        return {**initial, "errors": [error], "final_report": {
            "status": "workflow_error", "disclaimer": EDUCATIONAL_DISCLAIMER,
            "markdown": f"> **{EDUCATIONAL_DISCLAIMER}**\n\nThe workflow failed: {error}",
        }}
