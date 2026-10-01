"""Safety Review Agent.

Deterministic, rule-based review of the reasoning agents' output. It is
intentionally NOT LLM-based so that the safety layer is predictable and
testable. It detects:
  * unsupported certainty       ("definitely", "confirmed diagnosis", ...)
  * treatment / medication advice ("prescribe", doses, drug names, ...)
  * unsafe language              ("no need to see a doctor", ...)
  * a missing educational disclaimer
It assigns a risk level, flags cases for human review, produces feedback for
the Diagnosis Agent, and can redact violating text.
"""

import copy
import re
from typing import Dict, Iterator, List, Tuple

from app.models.schemas import SafetyIssue, SafetyReviewOutput
from app.utils.validators import EDUCATIONAL_DISCLAIMER

REDACTION = "[removed by safety review]"

RULES: Dict[str, List[str]] = {
    "unsupported_certainty": [
        r"\bdefinitely\b", r"\bcertainly\b", r"\bundoubtedly\b", r"\bwithout (a )?doubt\b",
        r"\bconfirmed diagnosis\b", r"\bdiagnosis is confirmed\b", r"\b100\s?%",
        r"\bguaranteed?\b", r"\bthe patient has\b", r"\bis diagnosed with\b", r"\bclearly has\b",
        # claiming a step is required (the plan may only say what "could be considered")
        r"\b(should|must) be (measured|performed|done|obtained|ordered|carried out)\b",
        r"\b(is|are) (mandatory|essential|warranted|required)\b", r"\bstrongly supports? the diagnosis\b",
    ],
    "treatment_recommendation": [
        r"\bprescrib\w*", r"\btreatment plan\b",
        r"\b\d+(\.\d+)?\s?(mg|mcg|ml|units?|iu)\b(?!\s*/)",  # doses, but not lab units like mg/L
        r"\b(start|begin|initiate|commence)\w*\s+(on\s+)?(an?\s+)?(antibiotics?|antivirals?|insulin|steroids?|medication|treatment|therapy)\b",
        r"\b(take|administer|give)\s+(\w+\s+)?(antibiotics?|paracetamol|acetaminophen|ibuprofen|insulin|tablets?|medication|dose)\b",
        r"\b(increase|decrease|reduce|stop|discontinue|double)\s+(the\s+|their\s+|his\s+|her\s+)?(dose|dosage|medication)\b",
        r"\b(amoxicillin|azithromycin|ceftriaxone|doxycycline|oseltamivir|paracetamol|ibuprofen)\b",
    ],
    "unsafe_language": [
        r"\bno need to (see|consult|visit)\b", r"\b(ignore|dismiss)\s+(the\s+|these\s+)?symptoms?\b",
        r"\bself[- ]?medicat\w*", r"\bnothing to worry\b",
        r"\b(don't|do not) need (a |to see a )?(doctor|clinician|medical)\b",
        r"\bavoid (seeing|consulting|visiting) (a )?(doctor|clinician|hospital)\b",
    ],
}
COMPILED = {kind: [re.compile(p, re.IGNORECASE) for p in patterns] for kind, patterns in RULES.items()}

FEEDBACK_HINTS = {
    "unsupported_certainty": "Rephrase as a possibility to consider; do not claim certainty or that a step is required.",
    "treatment_recommendation": "Remove all medication/treatment content; suggest diagnostic steps only.",
    "unsafe_language": "Remove language that discourages clinician involvement.",
}


def iter_strings(obj, path: str = "") -> Iterator[Tuple[str, str]]:
    """Yield (location, text) for every string in a nested dict/list."""
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if key in ("disclaimer", "method", "llm_error"):
                continue  # metadata, not generated clinical content
            yield from iter_strings(value, f"{path}.{key}" if path else key)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from iter_strings(value, f"{path}[{i}]")


def scan_text(text: str) -> List[Tuple[str, str]]:
    """Return (issue_type, matched_excerpt) pairs found in text."""
    hits = []
    for kind, patterns in COMPILED.items():
        for pattern in patterns:
            for match in pattern.finditer(text):
                hits.append((kind, match.group(0)))
    return hits


def redact_text(text: str) -> str:
    for patterns in COMPILED.values():
        for pattern in patterns:
            text = pattern.sub(REDACTION, text)
    return text


def redact(obj):
    """Return a deep copy of obj with all violating phrases redacted."""
    if isinstance(obj, str):
        return redact_text(obj)
    if isinstance(obj, dict):
        return {k: (v if k in ("disclaimer", "method", "llm_error") else redact(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    return copy.deepcopy(obj)


class SafetyReviewAgent:
    def review(self, differential: dict, plan: dict, red_flags: List[str] = None,
               upstream_errors: List[str] = None) -> dict:
        issues: List[SafetyIssue] = []
        for section_name, section in (("differential_diagnosis", differential), ("diagnostic_plan", plan)):
            for location, text in iter_strings(section, section_name):
                for kind, excerpt in scan_text(text):
                    issues.append(SafetyIssue(issue_type=kind, excerpt=excerpt, location=location))
            if "educational simulation" not in str(section.get("disclaimer", "")).lower():
                issues.append(SafetyIssue(issue_type="missing_disclaimer", excerpt="", location=f"{section_name}.disclaimer"))

        kinds = {i.issue_type for i in issues}
        notes: List[str] = []
        if red_flags:
            notes.append("Case contains warning symptoms: " + ", ".join(red_flags) + ".")
        if upstream_errors:
            notes.append(f"{len(upstream_errors)} upstream agent error(s) occurred; output may be incomplete.")

        if kinds & {"treatment_recommendation", "unsafe_language"} or red_flags:
            risk = "high"
        elif kinds or upstream_errors:
            risk = "moderate"
        else:
            risk = "low"

        output = SafetyReviewOutput(
            passed=not issues,
            risk_level=risk,
            requires_human_review=(risk == "high"),
            issues=issues,
            notes=notes or ["No safety rule violations detected."],
        )
        feedback = [
            f"{i.issue_type} at {i.location}: '{i.excerpt}'. {FEEDBACK_HINTS[i.issue_type]}"
            for i in issues if i.issue_type in FEEDBACK_HINTS
        ]
        return {**output.model_dump(), "feedback": feedback}

    @staticmethod
    def sanitize(differential: dict, plan: dict) -> Tuple[dict, dict]:
        """Redact violating phrases and enforce the educational disclaimer."""
        clean_diff, clean_plan = redact(differential), redact(plan)
        clean_diff["disclaimer"] = EDUCATIONAL_DISCLAIMER
        clean_plan["disclaimer"] = EDUCATIONAL_DISCLAIMER
        return clean_diff, clean_plan
