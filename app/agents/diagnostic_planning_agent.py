"""Diagnostic Planning Agent.

Suggests diagnostic steps that COULD BE CONSIDERED next for the simulated
case, with a rationale for each. It never suggests medication or treatment
and never claims a step is medically required.
"""

import json
from typing import Dict, List, Optional

from app.agents.diagnosis_agent import build_case_context
from app.llm import invoke_structured
from app.models.schemas import DiagnosticPlanOutput, DiagnosticStep
from app.prompts import PLANNING_HUMAN, PLANNING_SYSTEM
from app.rag.knowledge_base import KnowledgeBase
from app.utils.validators import EDUCATIONAL_DISCLAIMER

MAX_STEPS = 8


class DiagnosticPlanningAgent:
    def __init__(self, llm=None, knowledge_base: Optional[KnowledgeBase] = None):
        self.llm = llm
        self.kb = knowledge_base or KnowledgeBase()

    @staticmethod
    def _red_flag_step(red_flags: List[str]) -> DiagnosticStep:
        return DiagnosticStep(
            step="Prompt in-person clinician review of the warning symptoms",
            rationale=("The case includes warning features (" + ", ".join(red_flags) + ") that, in real "
                       "settings, are usually evaluated by a clinician without delay."),
            related_candidates=[],
            priority="prompt clinician review",
        )

    def rule_based(self, context: dict, candidates: List[dict], invalid_labs: List[str]) -> DiagnosticPlanOutput:
        steps: List[DiagnosticStep] = []
        if context.get("red_flags"):
            steps.append(self._red_flag_step(context["red_flags"]))

        by_text: Dict[str, DiagnosticStep] = {}
        for candidate in candidates:
            priority = "early" if candidate["consideration_level"] == "higher" else "routine"
            for ref in candidate.get("knowledge_refs", []):
                doc = self.kb.get(ref)
                if not doc:
                    continue
                for item in doc["general_workup"]:
                    if item in by_text:
                        by_text[item].related_candidates.append(candidate["condition"])
                        continue
                    by_text[item] = DiagnosticStep(
                        step=item,
                        rationale=f"Educational reference material lists this when evaluating {candidate['condition'].lower()}.",
                        related_candidates=[candidate["condition"]],
                        priority=priority,
                    )
        verify_step = []
        if invalid_labs:
            verify_step.append(DiagnosticStep(
                step="Verify or repeat the invalid laboratory entries: " + ", ".join(invalid_labs),
                rationale="These values were missing, non-numeric or negative and were excluded from the analysis.",
                priority="routine",
            ))

        # keep the red-flag and data-quality steps; trim work-up items to fit the cap
        room = MAX_STEPS - len(steps) - len(verify_step)
        steps = steps + list(by_text.values())[:room] + verify_step
        summary = (f"{len(steps)} diagnostic step(s) that could be considered for this simulated case."
                   if steps else "No diagnostic steps could be derived from the available information.")
        return DiagnosticPlanOutput(steps=steps, summary=summary)

    def _llm_plan(self, context: dict, candidates: List[dict], feedback: List[str]):
        workup = {}
        for candidate in candidates:
            for ref in candidate.get("knowledge_refs", []):
                doc = self.kb.get(ref)
                if doc:
                    workup[doc["title"]] = doc["general_workup"]
        messages = [
            ("system", PLANNING_SYSTEM),
            ("human", PLANNING_HUMAN.format(
                case_context=json.dumps(context, indent=2),
                candidates=json.dumps([{k: c.get(k) for k in ("condition", "consideration_level", "supporting_evidence")}
                                       for c in candidates], indent=2),
                workup=json.dumps(workup, indent=2) or "{}",
                safety_feedback="\n".join(f"- {f}" for f in feedback) or "None",
            )),
        ]
        result, error = invoke_structured(self.llm, DiagnosticPlanOutput, messages)
        if result is None:
            return None, error
        # Guarantee a real clinician-review step, first, whatever the LLM wrote
        # (a small model may label "ECG" as "prompt clinician review").
        if context.get("red_flags") and not any("clinician" in s.step.lower() for s in result.steps):
            result.steps.insert(0, self._red_flag_step(context["red_flags"]))
        result.steps = result.steps[:MAX_STEPS]
        result.disclaimer = EDUCATIONAL_DISCLAIMER
        return result, None

    def plan(self, symptom_analysis: dict, history_analysis: dict, lab_analysis: dict,
             differential: dict, safety_feedback: Optional[List[str]] = None) -> dict:
        context = build_case_context(symptom_analysis, history_analysis, lab_analysis)
        candidates = differential.get("candidates", [])
        output, llm_error, method = None, None, "rule-based"

        if self.llm is not None:
            output, llm_error = self._llm_plan(context, candidates, safety_feedback or [])
            if output is not None:
                method = "llm"
        if output is None:
            output = self.rule_based(context, candidates, lab_analysis.get("invalid_values", []))

        return {**output.model_dump(), "method": method, "llm_error": llm_error}
