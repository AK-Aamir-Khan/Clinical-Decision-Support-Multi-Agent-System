"""LangGraph node functions.

Each node wraps one agent: it reads the relevant part of the shared state,
calls the agent, and returns only the keys it owns plus a trace entry
("message") describing what it passed on to the next agent.
"""

import time
from typing import Callable, Dict, List

from app.agents.history_agent import MedicalHistoryAgent
from app.agents.lab_agent import LaboratoryAnalysisAgent
from app.agents.patient_agent import PatientDataAgent
from app.agents.symptom_agent import SymptomAnalysisAgent
from app.models.schemas import PatientCase
from app.utils.validators import sanitize_error
from app.workflow.state import ClinicalState

NodeFn = Callable[[ClinicalState], dict]


def traced(agent_name: str, reads: List[str], writes: List[str], fallback: Callable[[], dict]):
    """Decorator: time the node, record a trace message, and turn any
    unexpected exception into a safe fallback plus a sanitised error."""

    def decorator(fn: NodeFn) -> NodeFn:
        def wrapper(state: ClinicalState) -> dict:
            start = time.perf_counter()
            try:
                update = fn(state)
                status, message = "ok", update.pop("_message", "completed")
                errors: List[str] = update.pop("_errors", [])
            except Exception as exc:  # noqa: BLE001 - keep the workflow alive
                update = fallback()
                status = "error"
                message = "failed; fallback output used"
                errors = [f"{agent_name}: {sanitize_error(exc)}"]
            update["agent_trace"] = [{
                "agent": agent_name,
                "status": status,
                "reads": reads,
                "writes": writes,
                "message": message,
                "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            }]
            update["errors"] = errors
            return update
        wrapper.__name__ = fn.__name__
        return wrapper

    return decorator


def patient_from_state(state: ClinicalState) -> PatientCase:
    return PatientCase(**state["patient"])


def build_analysis_nodes() -> Dict[str, NodeFn]:
    """Nodes for the deterministic (non-LLM) agents."""
    patient_agent = PatientDataAgent()
    symptom_agent = SymptomAnalysisAgent()
    history_agent = MedicalHistoryAgent()
    lab_agent = LaboratoryAnalysisAgent()

    @traced("Patient Data Agent", ["patient_input"], ["patient", "validation"],
            lambda: {"patient": None, "validation": {"valid": False, "issues": ["Patient processing failed."]}})
    def patient_data_node(state: ClinicalState) -> dict:
        result = patient_agent.process_raw(state.get("patient_input", {}))
        patient = result.pop("patient")
        validation = result["validation"]
        verdict = "valid" if validation["valid"] else f"invalid ({len(validation['issues'])} issue(s))"
        return {
            "patient": patient.model_dump() if patient else None,
            "validation": validation,
            "_message": f"Patient record {verdict}; structured case shared with analysis agents.",
        }

    @traced("Symptom Analysis Agent", ["patient"], ["symptom_analysis"],
            lambda: {"symptom_analysis": {"analysis": "Symptom analysis unavailable.", "key_features": [], "red_flags": []}})
    def symptom_node(state: ClinicalState) -> dict:
        result = symptom_agent.analyze(patient_from_state(state))
        return {"symptom_analysis": result,
                "_message": f"{result['symptom_count']} symptom(s), systems: {', '.join(result['body_systems']) or 'none mapped'}."}

    @traced("Medical History Agent", ["patient"], ["history_analysis"],
            lambda: {"history_analysis": {"summary": "History analysis unavailable.", "conditions": [], "risk_factors": []}})
    def history_node(state: ClinicalState) -> dict:
        result = history_agent.analyze(patient_from_state(state))
        return {"history_analysis": result, "_message": result["summary"]}

    @traced("Laboratory Analysis Agent", ["patient"], ["lab_analysis"],
            lambda: {"lab_analysis": {"summary": "Lab analysis unavailable.", "results": [], "abnormal_findings": []}})
    def lab_node(state: ClinicalState) -> dict:
        result = lab_agent.analyze(patient_from_state(state))
        return {"lab_analysis": result, "_message": result["summary"]}

    return {
        "patient_data": patient_data_node,
        "symptom_analysis": symptom_node,
        "history_analysis": history_node,
        "lab_analysis": lab_node,
    }


def build_reasoning_nodes(llm=None, settings=None) -> Dict[str, NodeFn]:
    """Nodes for the RAG layer, LLM-backed reasoning agents, safety and report."""
    from app.agents.diagnosis_agent import DifferentialDiagnosisAgent
    from app.agents.diagnostic_planning_agent import DiagnosticPlanningAgent
    from app.agents.report_agent import ReportGenerator
    from app.agents.safety_agent import SafetyReviewAgent
    from app.config import get_settings
    from app.rag.knowledge_base import KnowledgeBase
    from app.rag.retriever import Retriever, build_retrieval_query

    settings = settings or get_settings()
    kb = KnowledgeBase()
    retriever = Retriever(kb)
    diagnosis_agent = DifferentialDiagnosisAgent(llm=llm, knowledge_base=kb)
    planning_agent = DiagnosticPlanningAgent(llm=llm, knowledge_base=kb)
    safety_agent = SafetyReviewAgent()
    report_generator = ReportGenerator()

    @traced("RAG Retriever", ["symptom_analysis", "history_analysis", "lab_analysis"],
            ["retrieval_query", "retrieved_knowledge"],
            lambda: {"retrieval_query": "", "retrieved_knowledge": []})
    def retrieval_node(state: ClinicalState) -> dict:
        query = build_retrieval_query(state.get("symptom_analysis", {}), state.get("history_analysis", {}),
                                      state.get("lab_analysis", {}))
        docs = retriever.retrieve(query["query"], top_k=settings.rag_top_k, lab_signals=query["lab_signals"])
        message = (f"Retrieved {len(docs)} document(s): " + ", ".join(d["id"] for d in docs)
                   if docs else "No relevant educational knowledge retrieved.")
        return {"retrieval_query": query["query"], "retrieved_knowledge": docs, "_message": message}

    @traced("Differential Diagnosis Agent",
            ["symptom_analysis", "history_analysis", "lab_analysis", "retrieved_knowledge", "safety_feedback"],
            ["differential_diagnosis"],
            lambda: {"differential_diagnosis": {"candidates": [], "reasoning_summary": "Unavailable.",
                                                "disclaimer": "", "method": "none", "llm_error": None}})
    def diagnosis_node(state: ClinicalState) -> dict:
        result = diagnosis_agent.diagnose(
            state.get("symptom_analysis", {}), state.get("history_analysis", {}), state.get("lab_analysis", {}),
            state.get("retrieved_knowledge", []), state.get("safety_feedback", []))
        errors = [f"Differential Diagnosis Agent LLM fallback: {result['llm_error']}"] if result["llm_error"] else []
        return {"differential_diagnosis": result, "_errors": errors,
                "_message": f"{len(result['candidates'])} candidate(s) via {result['method']} reasoning"
                            + (f"; {len(result['guardrail_notes'])} guardrail adjustment(s)."
                               if result.get("guardrail_notes") else ".")}

    @traced("Diagnostic Planning Agent",
            ["symptom_analysis", "history_analysis", "lab_analysis", "differential_diagnosis", "safety_feedback"],
            ["diagnostic_plan"],
            lambda: {"diagnostic_plan": {"steps": [], "summary": "Unavailable.", "disclaimer": "",
                                         "method": "none", "llm_error": None}})
    def planning_node(state: ClinicalState) -> dict:
        result = planning_agent.plan(state.get("symptom_analysis", {}), state.get("history_analysis", {}),
                                     state.get("lab_analysis", {}), state.get("differential_diagnosis", {}),
                                     state.get("safety_feedback", []))
        errors = [f"Diagnostic Planning Agent LLM fallback: {result['llm_error']}"] if result["llm_error"] else []
        return {"diagnostic_plan": result, "_errors": errors,
                "_message": f"{len(result['steps'])} diagnostic step(s) via {result['method']} reasoning."}

    @traced("Safety Review Agent", ["differential_diagnosis", "diagnostic_plan", "symptom_analysis"],
            ["safety_review", "safety_feedback", "revision_count"],
            lambda: {"safety_review": {"passed": False, "risk_level": "high", "requires_human_review": True,
                                       "issues": [], "notes": ["Safety review failed."], "feedback": [],
                                       "action": "finalize"}})
    def safety_node(state: ClinicalState) -> dict:
        differential = state.get("differential_diagnosis", {})
        plan = state.get("diagnostic_plan", {})
        review = safety_agent.review(differential, plan,
                                     red_flags=state.get("symptom_analysis", {}).get("red_flags", []),
                                     upstream_errors=state.get("errors", []))
        revisions = state.get("revision_count", 0)
        llm_generated = "llm" in (differential.get("method"), plan.get("method"))
        can_revise = llm_generated and review["feedback"] and revisions < settings.max_safety_revisions

        update = {"safety_review": review}
        if can_revise:
            review["action"] = "revise"
            update.update(safety_feedback=review["feedback"], revision_count=revisions + 1)
            message = f"{len(review['issues'])} issue(s); feedback sent back to Differential Diagnosis Agent."
        else:
            review["action"] = "finalize"
            if not review["passed"]:
                review["sanitized_differential"], review["sanitized_plan"] = safety_agent.sanitize(differential, plan)
            message = (f"Risk {review['risk_level']}; "
                       + ("passed." if review["passed"] else f"{len(review['issues'])} issue(s) redacted."))
        update["_message"] = message
        return update

    @traced("Report Generator", ["*"], ["final_report"],
            lambda: {"final_report": {"status": "error", "markdown": "Report generation failed."}})
    def report_node(state: ClinicalState) -> dict:
        report = report_generator.build(state)
        return {"final_report": report, "_message": f"Report status: {report['status']}."}

    return {
        "knowledge_retrieval": retrieval_node,
        "differential_diagnosis": diagnosis_node,
        "diagnostic_planning": planning_node,
        "safety_review": safety_node,
        "final_report": report_node,
    }
