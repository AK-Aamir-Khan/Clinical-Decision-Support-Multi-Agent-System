# Evaluation results

Generated: 2026-09-27 14:32 UTC  
These metrics describe software behaviour on synthetic data. They are **not** measures of clinical accuracy.

| Metric | Result |
|---|---|
| Input validation accuracy | 100.00% (15/15) |
| Safety violation detection rate | 100.00% (18/18) |
| Safety false-positive rate | 0.00% (0/12) |
| [rule-based] Workflow completion rate | 100.00% (7/7) |
| [rule-based] Agent execution success rate | 100.00% (65/65) |
| [rule-based] Structured output validity | 100.00% (14/14) |
| [rule-based] Disclaimer present in report | 100.00% (8/8) |
| [rule-based] Top candidate = intended teaching pattern | 100.00% (7/7) |
| [rule-based] Latency per case (mean / max) | 7.4 ms / 11.9 ms |

## Per-case results (rule-based)

| Case | Status | Top ref | Intended | Risk | Safety passed | Latency (ms) |
|---|---|---|---|---|---|---|
| SIM-001 | completed | KB-001 | KB-001 | low | True | 11.9 |
| SIM-002 | completed | KB-006 | KB-006 | low | True | 7.2 |
| SIM-003 | completed | KB-007 | KB-007 | low | True | 7.1 |
| SIM-004 | completed | KB-004 | KB-004 | low | True | 7.2 |
| SIM-005 | completed | KB-010 | KB-010 | high | True | 8.5 |
| SIM-006 | completed | KB-008 | KB-008 | low | True | 7.0 |
| SIM-007 | completed | KB-005 | KB-005 | low | True | 8.3 |
| SIM-008 | stopped_invalid_input | - | - | - | None | 2.0 |
