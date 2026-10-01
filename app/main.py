"""Streamlit UI for MedAgent-CDSS (educational simulation).

Run from the repository root:
    streamlit run app/main.py
"""

import json
import sys
from pathlib import Path

# `streamlit run app/main.py` puts app/ on sys.path; add the repo root instead.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.config import DATA_DIR, get_settings  # noqa: E402
from app.llm import describe_llm, get_llm  # noqa: E402
from app.utils.validators import EDUCATIONAL_DISCLAIMER  # noqa: E402
from app.workflow.graph import build_graph, run_case  # noqa: E402

st.set_page_config(page_title="MedAgent-CDSS", page_icon="🩺", layout="wide")


@st.cache_data
def load_cases() -> list:
    with open(DATA_DIR / "synthetic_patients.json", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_resource
def get_graph(use_llm: bool):
    settings = get_settings()
    llm = get_llm(settings) if use_llm else None
    return build_graph(llm=llm, settings=settings), llm


def section(number: int, title: str):
    st.subheader(f"{number}. {title}")


def render_results(state: dict):
    report = state.get("final_report", {})
    validation = state.get("validation", {})

    section(1, "Patient information")
    if state.get("patient"):
        st.json(state["patient"], expanded=True)
    else:
        st.json(state.get("patient_input", {}), expanded=False)

    section(2, "Validation result")
    if validation.get("valid"):
        st.success("Patient input is valid.")
    else:
        st.error("Patient input is invalid - the workflow stopped before analysis.")
        for issue in validation.get("issues", []):
            st.write(f"- {issue}")
        return

    col1, col2 = st.columns(2)
    with col1:
        section(3, "Symptom analysis")
        sym = state.get("symptom_analysis", {})
        st.write(sym.get("analysis"))
        st.write(f"**Time course:** {sym.get('duration_category')}")
        st.write(f"**Body systems:** {', '.join(sym.get('body_systems', [])) or '-'}")
        if sym.get("red_flags"):
            st.warning("Warning symptoms: " + ", ".join(sym["red_flags"]))
    with col2:
        section(4, "Medical history analysis")
        hist = state.get("history_analysis", {})
        st.write(hist.get("summary"))
        if hist.get("conditions"):
            st.dataframe(pd.DataFrame(hist["conditions"]), hide_index=True, width="stretch")
        if hist.get("medications"):
            st.dataframe(pd.DataFrame(hist["medications"]), hide_index=True, width="stretch")
        for flag in hist.get("history_flags", []):
            st.info(flag)

    section(5, "Laboratory analysis")
    labs = state.get("lab_analysis", {})
    st.write(labs.get("summary"))
    if labs.get("results"):
        cols = ["test", "value", "unit", "reference_range", "status", "note"]
        st.dataframe(pd.DataFrame(labs["results"]).reindex(columns=cols), hide_index=True, width="stretch")
    st.caption(labs.get("note", ""))

    section(6, "Retrieved educational context (RAG)")
    st.caption(f"Query: `{state.get('retrieval_query', '')}`")
    docs = state.get("retrieved_knowledge", [])
    if not docs:
        st.info("No relevant educational knowledge was retrieved.")
    for doc in docs:
        with st.expander(f"{doc['id']} - {doc['title']} (score {doc['score']})"):
            st.write(doc["content"])
            st.caption(f"Matched terms: {', '.join(doc['matched_terms']) or '-'} | "
                       f"Matched lab signals: {', '.join(doc['matched_lab_signals']) or '-'}")

    section(7, "Differential diagnosis candidates (conditions to consider)")
    diff = report.get("differential_diagnosis", [])
    st.caption(state.get("differential_diagnosis", {}).get("reasoning_summary", ""))
    for note in state.get("differential_diagnosis", {}).get("guardrail_notes", []):
        st.info(f"Guardrail: {note}")
    if not diff:
        st.info("No candidates could be derived.")
    for c in diff:
        with st.container(border=True):
            st.markdown(f"**{c['condition']}** - *{c['consideration_level']} consideration* ({c.get('category', '')})")
            st.write("Supporting: " + ("; ".join(c.get("supporting_evidence", [])) or "-"))
            if c.get("missing_or_conflicting_evidence"):
                st.write("Missing / conflicting: " + "; ".join(c["missing_or_conflicting_evidence"]))
            st.caption("Knowledge refs: " + ", ".join(c.get("knowledge_refs", [])))

    section(8, "Suggested diagnostic next steps")
    steps = report.get("diagnostic_plan", [])
    if steps:
        st.dataframe(pd.DataFrame(steps).reindex(columns=["priority", "step", "rationale", "related_candidates"]),
                     hide_index=True, width="stretch")

    section(9, "Safety review")
    safety = state.get("safety_review", {})
    risk = safety.get("risk_level", "unknown")
    {"low": st.success, "moderate": st.warning, "high": st.error}.get(risk, st.info)(
        f"Risk level: {risk} | Passed: {safety.get('passed')} | "
        f"Human review required: {safety.get('requires_human_review')} | Revisions: {state.get('revision_count', 0)}")
    for note in safety.get("notes", []):
        st.write(f"- {note}")
    if safety.get("issues"):
        st.dataframe(pd.DataFrame(safety["issues"]), hide_index=True, width="stretch")


def main():
    st.title("🩺 MedAgent-CDSS")
    st.caption("A Multi-Agent Clinical Decision Support System for Educational Simulation")
    st.error(f"⚠️ {EDUCATIONAL_DISCLAIMER} All patient cases are synthetic.")

    settings = get_settings()
    cases = load_cases()

    with st.sidebar:
        st.header("Case input")
        labels = [f"{c['patient_id']} - {c.get('description', '')}" for c in cases]
        choice = st.selectbox("Synthetic patient case", labels)
        selected = cases[labels.index(choice)]
        case_json = st.text_area("Case JSON (editable)", json.dumps(selected, indent=2), height=320,
                                 key=f"json_{selected['patient_id']}")
        use_llm = st.toggle("Use LLM reasoning", value=settings.llm_enabled, disabled=not settings.llm_enabled,
                            help="Needs OPENAI_API_KEY or a local server (OPENAI_BASE_URL, e.g. Ollama) in .env. "
                                 "Without one, rule-based reasoning is used.")
        st.caption(describe_llm(settings))
        if settings.is_local:
            st.caption("Local models can take a minute or more per agent on a laptop CPU.")
        run = st.button("Run multi-agent analysis", type="primary", width="stretch")

    if run:
        try:
            patient_input = json.loads(case_json)
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc.msg} (line {exc.lineno})")
            return
        graph, llm = get_graph(use_llm)
        with st.spinner("Agents are analysing the synthetic case..."):
            state = run_case(patient_input, llm=llm, graph=graph)
        st.session_state["state"] = state

    state = st.session_state.get("state")
    if not state:
        st.info("Select a synthetic case in the sidebar and click **Run multi-agent analysis**.")
        return

    tab_results, tab_report, tab_trace = st.tabs(["Agent outputs", "10. Final educational report", "Agent trace"])
    with tab_results:
        render_results(state)
    with tab_report:
        report = state.get("final_report", {})
        st.markdown(report.get("markdown", "No report generated."))
        st.download_button("Download report (JSON)", json.dumps(report, indent=2, default=str),
                           file_name=f"report_{report.get('patient_id', 'case')}.json", mime="application/json")
    with tab_trace:
        st.caption("Messages passed between agents through the shared LangGraph state, in execution order.")
        trace_cols = ["agent", "status", "message", "reads", "writes", "duration_ms"]
        st.dataframe(pd.DataFrame(state.get("agent_trace", [])).reindex(columns=trace_cols),
                     hide_index=True, width="stretch")
        for error in state.get("errors", []):
            st.warning(error)


main()
