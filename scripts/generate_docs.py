"""Regenerate documentation artefacts from the actual code.

    python scripts/generate_docs.py

Produces (in docs/):
  workflow_graph.mmd   Mermaid diagram exported from the compiled LangGraph
  sample_input.json    a synthetic case (SIM-001)
  sample_output.json   the real final report + agent trace for that case (rule-based mode)
  architecture.png     \\ drawn with matplotlib (optional dev dependency:
  workflow.png         /  pip install -r requirements-dev.txt)
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("USE_LLM", "false")  # deterministic sample output

from app.config import DATA_DIR  # noqa: E402
from app.workflow.graph import build_graph, run_case  # noqa: E402

DOCS = ROOT / "docs"


def export_mermaid_and_samples():
    graph = build_graph()
    (DOCS / "workflow_graph.mmd").write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")

    case = json.loads((DATA_DIR / "synthetic_patients.json").read_text(encoding="utf-8"))[0]
    state = run_case(case, graph=graph)
    (DOCS / "sample_input.json").write_text(json.dumps(case, indent=2), encoding="utf-8")
    sample = {k: state.get(k) for k in ("final_report", "agent_trace", "safety_review", "retrieved_knowledge")}
    (DOCS / "sample_output.json").write_text(json.dumps(sample, indent=2, default=str), encoding="utf-8")


def _box(ax, x, y, text, w=0.22, h=0.055, color="#e8f0fe", edge="#3b5b9a", fontsize=9, bold=False):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.008",
                                facecolor=color, edgecolor=edge, linewidth=1.2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, weight="bold" if bold else "normal")


def _arrow(ax, x1, y1, x2, y2, text="", color="#333333", style="-|>", rad=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=1.3, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x1 + x2) / 2 + 0.01, (y1 + y2) / 2, text, fontsize=7.5, color=color, va="center")


def draw_workflow():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 12))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("MedAgent-CDSS - LangGraph workflow", fontsize=13, weight="bold")

    agent, rag, safe, io = "#e8f0fe", "#fff4d6", "#fde2e1", "#e6f4ea"
    _box(ax, 0.5, 0.95, "Synthetic patient case (JSON)", color=io)
    _box(ax, 0.5, 0.86, "Patient Data Agent\nvalidate + structure", color=agent)
    _box(ax, 0.2, 0.73, "Symptom Analysis\nAgent", color=agent)
    _box(ax, 0.5, 0.73, "Medical History\nAgent", color=agent)
    _box(ax, 0.8, 0.73, "Laboratory Analysis\nAgent", color=agent)
    _box(ax, 0.5, 0.61, "RAG Retriever\nquery -> TF-IDF -> top-k docs", w=0.3, color=rag)
    _box(ax, 0.5, 0.49, "Differential Diagnosis Agent\n(LLM or rule-based)", w=0.3, color=agent)
    _box(ax, 0.5, 0.37, "Diagnostic Planning Agent\n(LLM or rule-based)", w=0.3, color=agent)
    _box(ax, 0.5, 0.25, "Safety Review Agent\n(deterministic rules)", w=0.3, color=safe)
    _box(ax, 0.5, 0.11, "Final Educational Report\n+ disclaimer", w=0.3, color=io)

    _arrow(ax, 0.5, 0.92, 0.5, 0.89)
    for x in (0.2, 0.5, 0.8):
        _arrow(ax, 0.5, 0.83, x, 0.76)
        _arrow(ax, x, 0.70, 0.5, 0.64)
    ax.text(0.53, 0.80, "valid -> parallel fan-out", fontsize=7.5, color="#1a7f37")
    ax.text(0.53, 0.665, "fan-in (waits for all 3)", fontsize=7.5)
    _arrow(ax, 0.5, 0.58, 0.5, 0.52)
    _arrow(ax, 0.5, 0.46, 0.5, 0.40)
    _arrow(ax, 0.5, 0.34, 0.5, 0.28)
    _arrow(ax, 0.5, 0.22, 0.5, 0.14, "finalize (sanitise if needed)")
    _arrow(ax, 0.35, 0.25, 0.35, 0.49, color="#b3261e", rad=-0.5)
    ax.text(0.02, 0.37, "feedback loop:\nissues in LLM output\n-> revise (max 1)", fontsize=7.5, color="#b3261e")

    # invalid input path
    ax.plot([0.61, 0.95, 0.95], [0.86, 0.86, 0.11], color="#b3261e", lw=1.2)
    _arrow(ax, 0.95, 0.11, 0.65, 0.11, color="#b3261e")
    ax.text(0.955, 0.5, "invalid input -> stop", rotation=90, fontsize=7.5, color="#b3261e", va="center")

    ax.text(0.5, 0.03, "All agents read from / write to one shared ClinicalState (TypedDict).",
            ha="center", fontsize=8, style="italic")
    fig.savefig(DOCS / "workflow.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_architecture():
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("MedAgent-CDSS - system architecture", fontsize=13, weight="bold")

    _box(ax, 0.5, 0.91, "Streamlit UI  (app/main.py)  -  select / edit synthetic case, view every agent's output",
         w=0.9, h=0.07, color="#e6f4ea", fontsize=10)
    ax.add_patch(FancyBboxPatch((0.04, 0.33), 0.92, 0.47, boxstyle="round,pad=0.01",
                                facecolor="#f7f9fc", edgecolor="#3b5b9a", linewidth=1.5, linestyle="--"))
    ax.text(0.06, 0.775, "LangGraph StateGraph  (app/workflow/)  -  shared ClinicalState",
            fontsize=10, weight="bold", color="#3b5b9a")
    row1 = ["Patient Data\nAgent", "Symptom Analysis\nAgent", "Medical History\nAgent", "Laboratory Analysis\nAgent"]
    for i, name in enumerate(row1):
        _box(ax, 0.16 + i * 0.23, 0.66, name, w=0.19, h=0.09)
    row2 = [("RAG\nRetriever", "#fff4d6"), ("Differential\nDiagnosis Agent", "#e8f0fe"),
            ("Diagnostic\nPlanning Agent", "#e8f0fe"), ("Safety Review\nAgent", "#fde2e1"),
            ("Report\nGenerator", "#e6f4ea")]
    for i, (name, color) in enumerate(row2):
        _box(ax, 0.14 + i * 0.18, 0.44, name, w=0.15, h=0.09, color=color)
    _arrow(ax, 0.5, 0.875, 0.5, 0.80, "run_case(patient_input)")

    resources = [
        ("synthetic_patients.json", "used by: UI / Patient Data Agent", 0.12),
        ("lab_reference_ranges.json", "used by: Laboratory Agent", 0.31),
        ("knowledge_base.json\n(educational)", "used by: RAG Retriever,\nDiagnosis & Planning", 0.5),
        ("OpenAI LLM (optional)\nlangchain-openai", "used by: Diagnosis &\nPlanning Agents", 0.69),
        ("Pydantic schemas\n(structured output)", "used by: all agents", 0.88),
    ]
    for name, used_by, x in resources:
        _box(ax, x, 0.17, name, w=0.17, h=0.09, color="#eeeeee", edge="#666666")
        ax.text(x, 0.085, used_by, ha="center", va="center", fontsize=7.5, color="#555555")
        _arrow(ax, x, 0.32, x, 0.22, color="#777777")
    ax.text(0.5, 0.02, "LLM is optional: without OPENAI_API_KEY (or on any API/parse failure) the reasoning agents "
            "use transparent rule-based logic.", ha="center", fontsize=8.5, style="italic")
    fig.savefig(DOCS / "architecture.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    export_mermaid_and_samples()
    print("Wrote docs/workflow_graph.mmd, docs/sample_input.json, docs/sample_output.json")
    try:
        draw_workflow()
        draw_architecture()
        print("Wrote docs/workflow.png, docs/architecture.png")
    except ImportError:
        print("matplotlib not installed - skipped PNG diagrams (pip install -r requirements-dev.txt)")
