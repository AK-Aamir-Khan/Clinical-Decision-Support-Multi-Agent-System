"""Basic evaluation of MedAgent-CDSS on the synthetic cases.

Run from the repository root:
    python -m app.evaluation            # rule-based mode (no API key needed)
    python -m app.evaluation --llm      # also evaluates LLM mode (OpenAI key or local server)

Writes evaluation/results.json and evaluation/results.md.
Metrics describe software behaviour (validation, completion, structure, safety
rule coverage, latency). They are NOT measures of clinical accuracy.
"""

import argparse
import json
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone

from app.agents.safety_agent import scan_text
from app.config import DATA_DIR, PROJECT_ROOT, get_settings
from app.llm import get_llm
from app.models.schemas import DiagnosticPlanOutput, DifferentialDiagnosisOutput
from app.utils.validators import EDUCATIONAL_DISCLAIMER
from app.workflow.graph import build_graph, run_case

EVAL_DIR = PROJECT_ROOT / "evaluation"

# The knowledge-base pattern each synthetic case was *designed* to illustrate.
# This checks consistency with the case author's intent, not clinical accuracy.
INTENDED_PATTERN = {
    "SIM-001": "KB-001", "SIM-002": "KB-006", "SIM-003": "KB-007", "SIM-004": "KB-004",
    "SIM-005": "KB-010", "SIM-006": "KB-008", "SIM-007": "KB-005",
}

# Extra malformed inputs for the validation metric: (input, expected_valid)
VALIDATION_PROBES = [
    ({"patient_id": "P1", "age": 40, "sex": "Female", "symptoms": ["cough"]}, True),
    ({"patient_id": "P2", "age": 40, "sex": "Female", "symptoms": []}, False),
    ({"patient_id": "P3", "age": -1, "sex": "Male", "symptoms": ["fever"]}, False),
    ({"age": 40, "sex": "Male", "symptoms": ["fever"]}, False),
    ({"patient_id": "P5", "age": "forty", "sex": "Male", "symptoms": ["fever"]}, False),
    ({"patient_id": "P6", "age": 40, "sex": "Male", "symptoms": ["fever"], "lab_results": {"WBC": "high"}}, False),
    ({"patient_id": "P7", "age": 40, "sex": "", "symptoms": ["fever"]}, False),
]


def _rate(numerator: int, denominator: int) -> dict:
    return {"value": round(numerator / denominator, 4) if denominator else None,
            "numerator": numerator, "denominator": denominator}


def evaluate_mode(cases: list, llm=None) -> dict:
    graph = build_graph(llm=llm)
    per_case, latencies = [], []
    agent_ok = agent_total = completed = expected_completed = 0
    structured_ok = structured_total = disclaimers = reports = 0
    pattern_hits = pattern_total = 0
    agent_times = defaultdict(list)

    for case in cases:
        start = time.perf_counter()
        state = run_case(case, llm=llm, graph=graph)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)
        report = state.get("final_report", {})
        valid = state.get("validation", {}).get("valid", False)

        for entry in state.get("agent_trace", []):
            agent_total += 1
            agent_ok += entry["status"] == "ok"
            agent_times[entry["agent"]].append(entry["duration_ms"])

        reports += 1
        disclaimers += EDUCATIONAL_DISCLAIMER in report.get("markdown", "")

        top_ref = None
        if valid:
            expected_completed += 1
            completed += report.get("status") == "completed"
            for key, schema in (("differential_diagnosis", DifferentialDiagnosisOutput),
                                ("diagnostic_plan", DiagnosticPlanOutput)):
                structured_total += 1
                try:
                    schema.model_validate(state.get(key, {}))
                    structured_ok += 1
                except Exception:  # noqa: BLE001
                    pass
            candidates = state.get("differential_diagnosis", {}).get("candidates", [])
            refs = candidates[0].get("knowledge_refs", []) if candidates else []
            top_ref = refs[0] if refs else None
            if case["patient_id"] in INTENDED_PATTERN:
                pattern_total += 1
                pattern_hits += top_ref == INTENDED_PATTERN[case["patient_id"]]

        per_case.append({
            "patient_id": case.get("patient_id"),
            "valid_input": valid,
            "status": report.get("status"),
            "top_candidate_ref": top_ref,
            "intended_pattern": INTENDED_PATTERN.get(case.get("patient_id")),
            "risk_level": state.get("safety_review", {}).get("risk_level"),
            "safety_passed": state.get("safety_review", {}).get("passed"),
            "revisions": state.get("revision_count", 0),
            "errors": len(state.get("errors", [])),
            "latency_ms": round(elapsed, 1),
        })

    return {
        "workflow_completion_rate": _rate(completed, expected_completed),
        "agent_execution_success_rate": _rate(agent_ok, agent_total),
        "structured_output_validity": _rate(structured_ok, structured_total),
        "disclaimer_presence_rate": _rate(disclaimers, reports),
        "top_candidate_matches_intended_pattern": _rate(pattern_hits, pattern_total),
        "latency_ms": {"mean": round(statistics.mean(latencies), 1), "max": round(max(latencies), 1),
                       "note": "Wall-clock time per case on the machine that ran the evaluation."},
        "mean_agent_latency_ms": {a: round(statistics.mean(t), 2) for a, t in agent_times.items()},
        "per_case": per_case,
    }


def evaluate_validation(cases: list) -> dict:
    from app.agents.patient_agent import PatientDataAgent

    agent = PatientDataAgent()
    probes = [(c, c["patient_id"] != "SIM-008") for c in cases] + VALIDATION_PROBES
    correct = sum(agent.process_raw(data)["validation"]["valid"] == expected for data, expected in probes)
    return _rate(correct, len(probes))


def evaluate_safety_rules() -> dict:
    data = json.loads((EVAL_DIR / "safety_test_set.json").read_text(encoding="utf-8"))
    detected = [text for text in data["unsafe"] if scan_text(text)]
    false_pos = [text for text in data["safe"] if scan_text(text)]
    return {
        "violation_detection_rate": _rate(len(detected), len(data["unsafe"])),
        "false_positive_rate": _rate(len(false_pos), len(data["safe"])),
        "missed": [t for t in data["unsafe"] if t not in detected],
        "false_positives": false_pos,
        "note": "Measured on a small hand-written labelled set (evaluation/safety_test_set.json).",
    }


def to_markdown(results: dict) -> str:
    def fmt(metric):
        return f"{metric['value']:.2%} ({metric['numerator']}/{metric['denominator']})" \
            if metric.get("value") is not None else "n/a"

    lines = ["# Evaluation results", "",
             f"Generated: {results['generated_at']}  ",
             "These metrics describe software behaviour on synthetic data. They are **not** measures of clinical accuracy.",
             "", "| Metric | Result |", "|---|---|",
             f"| Input validation accuracy | {fmt(results['input_validation_accuracy'])} |",
             f"| Safety violation detection rate | {fmt(results['safety_rules']['violation_detection_rate'])} |",
             f"| Safety false-positive rate | {fmt(results['safety_rules']['false_positive_rate'])} |"]
    for mode, res in results["modes"].items():
        if "skipped" in res:
            lines.append(f"| [{mode}] | not measured: {res['skipped']} |")
            continue
        lines += [
            f"| [{mode}] Workflow completion rate | {fmt(res['workflow_completion_rate'])} |",
            f"| [{mode}] Agent execution success rate | {fmt(res['agent_execution_success_rate'])} |",
            f"| [{mode}] Structured output validity | {fmt(res['structured_output_validity'])} |",
            f"| [{mode}] Disclaimer present in report | {fmt(res['disclaimer_presence_rate'])} |",
            f"| [{mode}] Top candidate = intended teaching pattern | {fmt(res['top_candidate_matches_intended_pattern'])} |",
            f"| [{mode}] Latency per case (mean / max) | {res['latency_ms']['mean']} ms / {res['latency_ms']['max']} ms |",
        ]
    for mode, res in results["modes"].items():
        if "per_case" not in res:
            continue
        lines += ["", f"## Per-case results ({mode})", "",
                  "| Case | Status | Top ref | Intended | Risk | Safety passed | Latency (ms) |", "|---|---|---|---|---|---|---|"]
        for c in res["per_case"]:
            lines.append(f"| {c['patient_id']} | {c['status']} | {c['top_candidate_ref'] or '-'} | "
                         f"{c['intended_pattern'] or '-'} | {c['risk_level'] or '-'} | {c['safety_passed']} | {c['latency_ms']} |")
    return "\n".join(lines) + "\n"


def run_evaluation(include_llm: bool = False, write_files: bool = True) -> dict:
    cases = json.loads((DATA_DIR / "synthetic_patients.json").read_text(encoding="utf-8"))
    results = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "num_cases": len(cases),
        "input_validation_accuracy": evaluate_validation(cases),
        "safety_rules": evaluate_safety_rules(),
        "modes": {"rule-based": evaluate_mode(cases, llm=None)},
    }
    if include_llm:
        llm = get_llm(get_settings())
        results["modes"]["llm"] = (evaluate_mode(cases, llm=llm) if llm is not None
                                   else {"skipped": "no LLM configured (OPENAI_API_KEY or OPENAI_BASE_URL)"})
    if write_files:
        EVAL_DIR.mkdir(exist_ok=True)
        (EVAL_DIR / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        (EVAL_DIR / "results.md").write_text(to_markdown(results), encoding="utf-8")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MedAgent-CDSS on synthetic cases.")
    parser.add_argument("--llm", action="store_true", help="Also evaluate LLM mode (OpenAI API or local server).")
    args = parser.parse_args()
    print(to_markdown(run_evaluation(include_llm=args.llm)))
