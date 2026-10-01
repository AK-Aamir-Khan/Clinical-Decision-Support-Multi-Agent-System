import json

import pytest

from app.rag.knowledge_base import KnowledgeBase
from app.rag.retriever import Retriever, build_retrieval_query, tokenize


@pytest.fixture(scope="module")
def retriever():
    return Retriever()


def test_knowledge_base_loads_and_is_labelled_educational():
    kb = KnowledgeBase()
    assert len(kb) >= 10
    assert "educational" in kb.meta["description"].lower()
    assert kb.get("KB-001")["title"] == "Community-acquired pneumonia"


def test_knowledge_base_rejects_incomplete_documents(tmp_path):
    bad = tmp_path / "kb.json"
    bad.write_text(json.dumps({"documents": [{"id": "X", "title": "t"}]}))
    with pytest.raises(ValueError):
        KnowledgeBase(bad)


def test_tokenize_removes_stopwords():
    assert tokenize("The Fever and the Cough") == ["fever", "cough"]


def test_respiratory_query_ranks_pneumonia_first(retriever):
    results = retriever.retrieve(
        "fever cough fatigue WBC high CRP high",
        lab_signals=[{"test": "WBC", "status": "high"}, {"test": "CRP", "status": "high"}],
    )
    assert results[0]["id"] == "KB-001"
    assert "WBC high" in results[0]["matched_lab_signals"]
    assert results == sorted(results, key=lambda r: r["score"], reverse=True)


def test_anemia_query(retriever):
    results = retriever.retrieve(
        "fatigue pallor dizziness",
        lab_signals=[{"test": "Hemoglobin", "status": "low"}, {"test": "Ferritin", "status": "low"}],
    )
    assert results[0]["id"] == "KB-006"


def test_empty_and_irrelevant_queries_return_empty(retriever):
    assert retriever.retrieve("") == []
    assert retriever.retrieve("zzzz qqqq") == []


def test_top_k_respected(retriever):
    assert len(retriever.retrieve("fever", top_k=2)) <= 2


def test_build_retrieval_query():
    q = build_retrieval_query(
        {"key_features": ["fever", "cough"]},
        {"conditions": [{"name": "hypertension"}]},
        {"results": [{"test": "WBC", "status": "high"}, {"test": "CRP", "status": "normal"}]},
    )
    assert "fever" in q["query"] and "hypertension" in q["query"] and "WBC high" in q["query"]
    assert q["lab_signals"] == [{"test": "WBC", "status": "high"}]
