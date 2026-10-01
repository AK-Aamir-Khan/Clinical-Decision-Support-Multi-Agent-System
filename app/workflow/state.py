"""Shared LangGraph state.

Every agent reads the fields it needs from this state and writes back only
its own section. This shared, structured state is how agents communicate.
"""

import operator
from typing import Annotated, List, Optional, TypedDict


class ClinicalState(TypedDict, total=False):
    # input
    patient_input: dict

    # Patient Data Agent
    patient: Optional[dict]
    validation: dict

    # parallel analysis agents
    symptom_analysis: dict
    history_analysis: dict
    lab_analysis: dict

    # RAG layer
    retrieval_query: str
    retrieved_knowledge: List[dict]

    # reasoning agents
    differential_diagnosis: dict
    diagnostic_plan: dict

    # Safety Review Agent (+ feedback loop to the diagnosis agent)
    safety_review: dict
    safety_feedback: List[str]
    revision_count: int

    # output
    final_report: dict
    llm_mode: str
    llm_model: Optional[str]

    # append-only logs: parallel branches can write to them safely
    agent_trace: Annotated[List[dict], operator.add]
    errors: Annotated[List[str], operator.add]
