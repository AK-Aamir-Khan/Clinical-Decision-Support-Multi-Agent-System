from app.evaluation import evaluate_safety_rules, run_evaluation, to_markdown


def test_evaluation_runs_and_reports_metrics():
    results = run_evaluation(include_llm=False, write_files=False)
    rule = results["modes"]["rule-based"]

    assert results["num_cases"] >= 6
    assert rule["workflow_completion_rate"]["denominator"] >= 5
    assert 0 <= rule["agent_execution_success_rate"]["value"] <= 1
    assert "Evaluation results" in to_markdown(results)


def test_safety_rule_metrics_structure():
    safety = evaluate_safety_rules()
    assert safety["violation_detection_rate"]["denominator"] > 0
    assert safety["false_positive_rate"]["denominator"] > 0
