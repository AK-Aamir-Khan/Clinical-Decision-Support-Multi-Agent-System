"""Differential Diagnosis Agent.

Consumes the structured outputs of the Symptom, History and Laboratory agents
plus the RAG context, and produces a list of conditions to CONSIDER with
evidence. Uses the LLM when configured; otherwise (or on any LLM failure)
falls back to transparent rule-based scoring over the knowledge base.
"""

import json
from typing import List, Optional

from app.llm import invoke_structured
from app.models.schemas import DiagnosisCandidate, DifferentialDiagnosisOutput
from app.prompts import DIAGNOSIS_HUMAN, DIAGNOSIS_SYSTEM
from app.rag.knowledge_base import KnowledgeBase
from app.utils.validators import EDUCATIONAL_DISCLAIMER, normalize_term

NON_CONDITION_CATEGORIES = {"laboratory interpretation"}
MAX_CANDIDATES = 5
MIN_POINTS = 1.5  # symptom match = 1, lab match = 1.5, history match = 0.5, conflict = -1


def _term_matches(keyword: str, feature: str) -> bool:
    keyword, feature = normalize_term(keyword), normalize_term(feature)
    return keyword == feature or keyword in feature or (len(feature) > 3 and feature in keyword)


def build_case_context(symptoms: dict, history: dict, labs: dict) -> dict:
    """Only the fields the reasoning agents need (keeps prompts small and focused)."""
    return {
        "symptoms": symptoms.get("key_features", []),
        "duration": symptoms.get("duration"),
        "duration_category": symptoms.get("duration_category"),
        "red_flags": symptoms.get("red_flags", []),
        "body_systems": symptoms.get("body_systems", []),
        "history_conditions": [c["name"] for c in history.get("conditions", [])],
        "risk_factors": history.get("risk_factors", []),
        "history_flags": history.get("history_flags", []),
        "abnormal_labs": labs.get("abnormal_findings", []),
        "lab_results": [
            {k: r.get(k) for k in ("test", "value", "unit", "status", "reference_range")}
            for r in labs.get("results", [])
        ],
    }


class DifferentialDiagnosisAgent:
    def __init__(self, llm=None, knowledge_base: Optional[KnowledgeBase] = None):
        self.llm = llm
        self.kb = knowledge_base or KnowledgeBase()

    # ------------------------------------------------------------------ rules
    @staticmethod
    def _evidence(doc: dict, context: dict, lab_results: List[dict]) -> dict:
        """Compare one knowledge document with the case: what matches, conflicts or is missing."""
        features = context["symptoms"]
        history = context["history_conditions"]
        measured = {r["test"]: r["status"] for r in lab_results}
        ev = {
            "symptom_hits": sorted({f for f in features for kw in doc["keywords"] if _term_matches(kw, f)}),
            "history_hits": sorted({h for h in history for kw in doc["keywords"] if _term_matches(kw, h)}),
            "red_flag_hits": sorted({f for f in context.get("red_flags", []) for kw in doc["keywords"]
                                     if _term_matches(kw, f)}),
            "lab_hits": [], "conflicts": [], "missing": [],
        }
        for signal in doc.get("lab_signals", []):
            test, expected = signal["test"], signal["status"]
            if test not in measured:
                ev["missing"].append(f"{test} not measured in this case")
            elif measured[test] == expected:
                ev["lab_hits"].append(f"{test} {expected}")
            elif measured[test] in ("low", "high", "normal"):
                ev["conflicts"].append(f"{test} is {measured[test]} (pattern often described: {expected})")
        return ev

    def _score_document(self, doc: dict, context: dict, lab_results: List[dict]) -> Optional[DiagnosisCandidate]:
        ev = self._evidence(doc, context, lab_results)
        symptom_hits, history_hits, lab_hits = ev["symptom_hits"], ev["history_hits"], ev["lab_hits"]
        conflicts, missing = ev["conflicts"], ev["missing"]

        points = len(symptom_hits) + 1.5 * len(lab_hits) + 0.5 * len(history_hits) - len(conflicts)
        if points < MIN_POINTS or not (symptom_hits or lab_hits):
            return None  # too little evidence to be worth listing
        if points >= 4 and len(symptom_hits) >= 2:  # labs alone never make a "higher" candidate
            level = "higher"
        elif points >= 2:
            level = "moderate"
        else:
            level = "lower"

        evidence = [f"Reported symptom: {s}" for s in symptom_hits]
        evidence += [f"Lab finding: {h}" for h in lab_hits]
        evidence += [f"History factor: {h}" for h in history_hits]
        return DiagnosisCandidate(
            condition=doc["title"],
            category=doc["category"],
            consideration_level=level,
            supporting_evidence=evidence,
            missing_or_conflicting_evidence=conflicts + missing,
            knowledge_refs=[doc["id"]],
        )

    def rule_based(self, context: dict, lab_results: List[dict], retrieved: List[dict]) -> DifferentialDiagnosisOutput:
        candidates = []
        for item in retrieved:
            doc = self.kb.get(item["id"])
            if not doc or doc["category"] in NON_CONDITION_CATEGORIES:
                continue
            candidate = self._score_document(doc, context, lab_results)
            if candidate:
                candidates.append(candidate)

        order = {"higher": 0, "moderate": 1, "lower": 2}
        candidates.sort(key=lambda c: (order[c.consideration_level], -len(c.supporting_evidence)))
        candidates = candidates[:MAX_CANDIDATES]

        if candidates:
            summary = (
                f"{len(candidates)} condition(s) to consider were derived by matching the case's "
                "symptoms, history and out-of-range labs against the retrieved educational knowledge. "
                "Levels are relative, not probabilities."
            )
        else:
            summary = ("Insufficient matching evidence in the educational knowledge base to suggest "
                       "conditions to consider for this simulated case.")
        return DifferentialDiagnosisOutput(candidates=candidates, reasoning_summary=summary)

    # ------------------------------------------------------------- guardrails
    def apply_guardrails(self, output: DifferentialDiagnosisOutput, context: dict, lab_results: List[dict],
                         from_llm: bool) -> List[str]:
        """Deterministic checks applied to every differential (LLM or rule-based).

        1. Grounding (LLM output only): a candidate must cite retrieved knowledge, and that
           document must match at least one reported symptom or out-of-range lab.
        2. Known history: a condition already documented in the history is context, not a new
           explanation for the presenting symptoms -> capped at "lower".
        3. Red flags: the candidate whose knowledge best matches the warning symptoms is
           listed first as "higher".
        Returns human-readable notes describing every change made.
        """
        notes: List[str] = []
        kept: List[DiagnosisCandidate] = []
        for cand in output.candidates:
            docs = [d for d in (self.kb.get(ref) for ref in cand.knowledge_refs) if d]
            if from_llm:
                if not docs:
                    notes.append(f"Removed '{cand.condition}': not linked to any retrieved knowledge.")
                    continue
                if not any(ev["symptom_hits"] or ev["lab_hits"]
                           for ev in (self._evidence(d, context, lab_results) for d in docs)):
                    notes.append(f"Removed '{cand.condition}': no reported symptom or abnormal lab "
                                 "matches the cited knowledge.")
                    continue
            if self._is_known_condition(cand.condition, context["history_conditions"]):
                if cand.consideration_level != "lower":
                    notes.append(f"'{cand.condition}' is already in the patient's history; set to lower.")
                cand.consideration_level = "lower"
                note = "Already documented in the history - context rather than a new explanation."
                if note not in cand.missing_or_conflicting_evidence:
                    cand.missing_or_conflicting_evidence.append(note)
            kept.append(cand)

        order = {"higher": 0, "moderate": 1, "lower": 2}
        kept.sort(key=lambda c: order[c.consideration_level])  # stable: keeps model order within a level
        if context.get("red_flags"):
            promoted = self._best_red_flag_match(kept, context, lab_results)
            if promoted is not None:
                if promoted.consideration_level != "higher" or kept[0] is not promoted:
                    notes.append(f"'{promoted.condition}' best explains the warning symptoms; listed first.")
                promoted.consideration_level = "higher"
                kept.remove(promoted)
                kept.insert(0, promoted)

        output.candidates = kept[:MAX_CANDIDATES]
        return notes

    @staticmethod
    def _is_known_condition(condition: str, history: List[str]) -> bool:
        cond = normalize_term(condition)
        return any(normalize_term(h) in cond or cond in normalize_term(h) for h in history if len(h) > 3)

    def _best_red_flag_match(self, candidates: List[DiagnosisCandidate], context: dict,
                             lab_results: List[dict]) -> Optional[DiagnosisCandidate]:
        best, best_score = None, 0
        for cand in candidates:
            for ref in cand.knowledge_refs:
                doc = self.kb.get(ref)
                if not doc:
                    continue
                ev = self._evidence(doc, context, lab_results)
                score = 10 * len(ev["red_flag_hits"]) + len(ev["symptom_hits"])  # red flags dominate
                if ev["red_flag_hits"] and score > best_score:
                    best, best_score = cand, score
        return best

    # -------------------------------------------------------------------- llm
    def _llm_diagnose(self, context: dict, retrieved: List[dict], feedback: List[str]):
        knowledge = [{"id": r["id"], "title": r["title"], "content": r["content"]} for r in retrieved]
        messages = [
            ("system", DIAGNOSIS_SYSTEM),
            ("human", DIAGNOSIS_HUMAN.format(
                case_context=json.dumps(context, indent=2),
                knowledge=json.dumps(knowledge, indent=2) if knowledge else "No relevant knowledge retrieved.",
                safety_feedback="\n".join(f"- {f}" for f in feedback) or "None",
            )),
        ]
        result, error = invoke_structured(self.llm, DifferentialDiagnosisOutput, messages)
        if result is None:
            return None, error
        allowed = {r["id"] for r in retrieved}
        for candidate in result.candidates:  # drop hallucinated references
            candidate.knowledge_refs = [ref for ref in candidate.knowledge_refs if ref in allowed]
        result.candidates = result.candidates[:MAX_CANDIDATES]
        result.disclaimer = EDUCATIONAL_DISCLAIMER
        return result, None

    # ------------------------------------------------------------------- main
    def diagnose(self, symptom_analysis: dict, history_analysis: dict, lab_analysis: dict,
                 retrieved: List[dict], safety_feedback: Optional[List[str]] = None) -> dict:
        context = build_case_context(symptom_analysis, history_analysis, lab_analysis)
        output, llm_error, method = None, None, "rule-based"

        if self.llm is not None:
            output, llm_error = self._llm_diagnose(context, retrieved, safety_feedback or [])
            if output is not None:
                method = "llm"
        if output is None:
            output = self.rule_based(context, lab_analysis.get("results", []), retrieved)

        notes = self.apply_guardrails(output, context, lab_analysis.get("results", []), from_llm=method == "llm")
        return {**output.model_dump(), "method": method, "llm_error": llm_error, "guardrail_notes": notes}
