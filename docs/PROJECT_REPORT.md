# MedAgent-CDSS - Notes for the Agentic AI Lab-I Report

> Educational simulation only. This system does not provide medical diagnosis, treatment, or professional medical advice.

Use these sections as the skeleton of the lab report. Figures referenced here are in `docs/` and `screenshots/`; numbers come from `evaluation/results.md` (regenerate with `python -m app.evaluation`).

## 1. Problem understanding

Clinical reasoning combines several distinct sub-tasks: gathering and checking data, interpreting symptoms, reviewing history, interpreting lab values against reference ranges, recalling relevant knowledge, generating a differential, planning investigations, and checking the result for safety. A single monolithic prompt mixes all of these, is hard to test, and can hallucinate. The project models each sub-task as an agent with a narrow responsibility and a structured output, so that each step can be inspected, tested and constrained. Because this is a teaching simulation, the system must never present its output as a diagnosis or give treatment advice.

## 2. Task decomposition

| Sub-task | Agent | Deterministic or LLM |
|---|---|---|
| Validate and structure the case | Patient Data Agent | Deterministic (Pydantic) |
| Interpret symptoms | Symptom Analysis Agent | Deterministic |
| Review history, medications, allergies | Medical History Agent | Deterministic |
| Compare labs to reference ranges | Laboratory Analysis Agent | Deterministic |
| Retrieve relevant knowledge | RAG Retriever | Deterministic (TF-IDF) |
| Generate conditions to consider | Differential Diagnosis Agent | LLM with rule-based fallback |
| Suggest diagnostic steps | Diagnostic Planning Agent | LLM with rule-based fallback |
| Check safety of generated text | Safety Review Agent | Deterministic (rules) |
| Assemble the report | Report Generator | Deterministic |

Design principle: use the LLM only where language generation adds value (reasoning over evidence), and keep data handling and safety deterministic.

## 3. Communication design

- **Shared state**: `ClinicalState` (`app/workflow/state.py`), a `TypedDict`. Each agent reads only the keys it needs and writes only its own keys (see the reads/writes table in the README).
- **Structured messages**: every agent output is a dict / Pydantic model, so the next agent consumes fields, not free text. The LLM receives only a compact `case_context` built from upstream outputs (`build_case_context`), never the whole raw state.
- **Parallelism**: Symptom, History and Lab agents run in parallel; `agent_trace` and `errors` use an `operator.add` reducer so parallel writes merge safely.
- **Conditional routing**: `route_after_validation` (valid → analyses, invalid → report) and `route_after_safety` (revise → diagnosis, finalize → report).
- **Feedback loop**: the Safety Review Agent writes `safety_feedback`, which the Differential Diagnosis Agent includes in its next prompt (max `MAX_SAFETY_REVISIONS`, default 1).
- **Trace**: each node appends `{agent, status, reads, writes, message, duration_ms}` to `agent_trace`, shown in the UI's *Agent trace* tab.

## 4. Tools used

LangGraph (orchestration), LangChain `ChatOpenAI` + `with_structured_output` (optional LLM), Pydantic (validation and output schemas), a pure-Python TF-IDF retriever over `data/knowledge_base.json`, JSON reference ranges, Streamlit (UI), pytest (tests), Docker (deployment), matplotlib (diagrams only, dev dependency).

## 5. Architecture and workflow

- Architecture: `docs/architecture.png`
- Workflow: `docs/workflow.png`
- Exact compiled graph: `docs/workflow_graph.mmd` (paste into https://mermaid.live to render)

## 6. Sample input / output

- Input: `docs/sample_input.json` (SIM-001)
- Output: `docs/sample_output.json` (final report, agent trace, safety review, retrieved knowledge) - produced by `scripts/generate_docs.py` from an actual run.

## 7. Agent interaction (example: SIM-001)

1. Patient Data Agent → "Patient record valid; structured case shared with analysis agents."
2. Symptom / History / Lab agents (parallel) → features, systems, conditions, WBC high, CRP high.
3. RAG Retriever → query `fever cough fatigue hypertension WBC high CRP high` → KB-001, KB-012, KB-005, KB-003.
4. Differential Diagnosis Agent → pneumonia (higher), UTI (moderate), influenza-like illness (moderate), each with evidence.
5. Diagnostic Planning Agent → chest X-ray, pulse oximetry, urinalysis, ... with rationale.
6. Safety Review Agent → risk low, passed.
7. Report Generator → final educational report.

For the safety loop, see `tests/test_full_workflow.py::test_safety_feedback_loop_revises_llm_output`, where a (mocked) LLM first writes "The patient definitely has pneumonia", the safety agent sends feedback, and the second draft passes.

## 8. Evaluation and testing

- `python -m pytest` - unit, integration and UI tests (LLM mocked, no API calls).
- `python -m app.evaluation` - metrics in `evaluation/results.md`.
- Limitations of the evaluation: cases and knowledge base were authored together with the rules; the safety set is small and hand-written; LLM-mode metrics require an API key and vary between runs.

## 8a. Running with a small local LLM (Gemma 3 4B)

The reasoning agents were tested with `gemma3:4b` via Ollama (JSON mode, because Gemma 3 in Ollama has no tool calling). Every reply was valid JSON and parsed with Pydantic (12-20 s per call on a laptop), and the SIM-001 answer was sensible. On the red-flag case SIM-005 the model ranked an already-known condition above the warning-symptom explanation, cited no knowledge for one candidate and invented a patient location as evidence for another; the planning reply used "is warranted" / "should be measured". The response was layered:

1. **Prompt engineering** - tighter constraints fixed the ranking but not the hallucination.
2. **Deterministic guardrails** after the LLM - grounding check (cited knowledge must match a reported symptom or abnormal lab), known-history cap, red-flag priority.
3. **Safety agent** - new "necessity" rules; feedback now reaches both the Diagnosis and Planning agents; residual violations are redacted.

Lesson for the report: a small local model is usable for structure and fluent explanation, but ranking and grounding must be enforced in code. The recorded replies are regression fixtures (`tests/data/gemma3_4b_recorded.json`).

## 9. Possible viva questions

- *Why LangGraph instead of calling functions in sequence?* Explicit shared state, parallel branches with safe merging, conditional routing and loops, and a graph that can be inspected/exported.
- *Why is the safety agent not an LLM?* A safety layer should be predictable and testable; an LLM critic could be added on top later.
- *What happens without an API key or if the API fails?* `invoke_structured` returns an error; the agent falls back to rule-based reasoning and the error is recorded (with secrets redacted).
- *How do you prevent hallucinated references?* `knowledge_refs` returned by the LLM are filtered to the IDs actually retrieved.
- *Why not trust the LLM's ranking?* With Gemma 3 4B it ranked a known history condition above a chest-pain presentation and invented evidence; guardrails in code enforce grounding and red-flag priority.
- *Why TF-IDF and not embeddings?* 13 short documents, keyword-rich; TF-IDF is reproducible, offline and explainable. Embeddings are listed as future work.
