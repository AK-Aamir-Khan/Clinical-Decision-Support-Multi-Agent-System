"""Report Generator: assembles the final educational report from the state."""

from typing import List

from app.utils.validators import EDUCATIONAL_DISCLAIMER


class ReportGenerator:
    def build(self, state: dict) -> dict:
        validation = state.get("validation", {})
        if not validation.get("valid"):
            report = {
                "status": "stopped_invalid_input",
                "disclaimer": EDUCATIONAL_DISCLAIMER,
                "patient_id": validation.get("patient_id"),
                "validation_issues": validation.get("issues", []),
                "errors": state.get("errors", []),
            }
            report["markdown"] = self._invalid_markdown(report)
            return report

        safety = state.get("safety_review", {})
        differential = state.get("differential_diagnosis", {})
        plan = state.get("diagnostic_plan", {})
        if safety and not safety.get("passed", True):
            differential = safety.get("sanitized_differential", differential)
            plan = safety.get("sanitized_plan", plan)

        patient = state.get("patient") or {}
        report = {
            "status": "completed",
            "disclaimer": EDUCATIONAL_DISCLAIMER,
            "patient_id": patient.get("patient_id"),
            "patient_summary": f"{patient.get('age')}-year-old {patient.get('sex', '').lower()} (synthetic case)",
            "llm_mode": self._reasoning_mode(state),
            "symptom_summary": state.get("symptom_analysis", {}).get("analysis"),
            "red_flags": state.get("symptom_analysis", {}).get("red_flags", []),
            "history_summary": state.get("history_analysis", {}).get("summary"),
            "lab_summary": state.get("lab_analysis", {}).get("summary"),
            "abnormal_labs": state.get("lab_analysis", {}).get("abnormal_findings", []),
            "knowledge_used": [f"{k['id']}: {k['title']}" for k in state.get("retrieved_knowledge", [])],
            "differential_diagnosis": differential.get("candidates", []),
            "diagnostic_plan": plan.get("steps", []),
            "safety": {
                "passed": safety.get("passed"),
                "risk_level": safety.get("risk_level"),
                "requires_human_review": safety.get("requires_human_review"),
                "issue_count": len(safety.get("issues", [])),
                "revisions": state.get("revision_count", 0),
            },
            "errors": state.get("errors", []),
        }
        report["markdown"] = self._markdown(report)
        return report

    @staticmethod
    def _reasoning_mode(state: dict) -> str:
        """Describe what actually produced the output, not just what was requested."""
        methods = {state.get("differential_diagnosis", {}).get("method"),
                   state.get("diagnostic_plan", {}).get("method")}
        model = f" ({state['llm_model']})" if state.get("llm_model") else ""
        if state.get("llm_mode") != "llm":
            return "rule-based"
        if methods == {"llm"}:
            return f"llm{model}"
        if "llm" in methods:
            return f"mixed: llm{model} + rule-based fallback"
        return f"rule-based fallback ({state.get('llm_model') or 'LLM'} unavailable)"

    @staticmethod
    def _invalid_markdown(report: dict) -> str:
        lines = [f"> **{EDUCATIONAL_DISCLAIMER}**", "", "## Workflow stopped: invalid patient input", ""]
        lines += [f"- {issue}" for issue in report["validation_issues"]]
        return "\n".join(lines)

    @staticmethod
    def _markdown(r: dict) -> str:
        lines: List[str] = [
            f"> **{r['disclaimer']}**", "",
            f"# Educational Case Report - {r['patient_id']}", "",
            f"**Patient:** {r['patient_summary']}  ",
            f"**Reasoning mode:** {r['llm_mode']}  ",
            f"**Safety risk level:** {r['safety']['risk_level']}"
            + (" - flagged for human review" if r["safety"]["requires_human_review"] else ""), "",
            "## Findings",
            f"- Symptoms: {r['symptom_summary']}",
            f"- Warning symptoms: {', '.join(r['red_flags']) or 'none identified'}",
            f"- History: {r['history_summary']}",
            f"- Labs: {r['lab_summary']}",
        ]
        if r["abnormal_labs"]:
            lines.append(f"- Out-of-range labs (reference comparison only): {', '.join(r['abnormal_labs'])}")
        lines += ["", "## Conditions to consider (not a diagnosis)"]
        if not r["differential_diagnosis"]:
            lines.append("- No candidates could be derived from the available information.")
        for c in r["differential_diagnosis"]:
            lines.append(f"- **{c['condition']}** ({c['consideration_level']} consideration): "
                         + "; ".join(c.get("supporting_evidence", [])))
        lines += ["", "## Diagnostic steps that could be considered"]
        if not r["diagnostic_plan"]:
            lines.append("- None derived.")
        for s in r["diagnostic_plan"]:
            lines.append(f"- [{s['priority']}] {s['step']} - {s['rationale']}")
        if r["knowledge_used"]:
            lines += ["", "## Educational knowledge used", *[f"- {k}" for k in r["knowledge_used"]]]
        return "\n".join(lines)
