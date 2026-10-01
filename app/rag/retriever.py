"""Simple, reproducible retriever for the educational knowledge base.

Scoring = TF-IDF cosine similarity between the query and each document
          + a small bonus for every matching structured lab signal
            (e.g. the patient's WBC is 'high' and the document lists WBC high).

Pure Python: no vector database, embeddings API or extra dependency.
"""

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional

from app.rag.knowledge_base import KnowledgeBase

_TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "be", "by", "can", "for", "from", "in", "is",
    "it", "of", "on", "or", "such", "that", "the", "to", "with", "which", "may",
    "often", "some", "this", "these", "their", "they", "when", "where", "into",
}
LAB_SIGNAL_BONUS = 0.1


def tokenize(text: str) -> List[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


class Retriever:
    def __init__(self, knowledge_base: Optional[KnowledgeBase] = None):
        self.kb = knowledge_base or KnowledgeBase()
        doc_tokens = [tokenize(KnowledgeBase.searchable_text(d)) for d in self.kb.documents]
        n_docs = len(doc_tokens)
        doc_freq = Counter(t for tokens in doc_tokens for t in set(tokens))
        self.idf: Dict[str, float] = {
            t: math.log((1 + n_docs) / (1 + df)) + 1 for t, df in doc_freq.items()
        }
        self.doc_vectors = [self._vectorize(tokens) for tokens in doc_tokens]

    def _vectorize(self, tokens: Iterable[str]) -> Dict[str, float]:
        counts = Counter(t for t in tokens if t in self.idf)
        vector = {t: c * self.idf[t] for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vector.values())) or 1.0
        return {t: v / norm for t, v in vector.items()}

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        lab_signals: Optional[List[dict]] = None,
        min_score: float = 0.05,
    ) -> List[dict]:
        """Return up to top_k documents ordered by relevance (may be empty)."""
        query_vector = self._vectorize(tokenize(query or ""))
        wanted = {(s["test"], s["status"]) for s in (lab_signals or [])}
        if not query_vector and not wanted:
            return []

        results = []
        for doc, doc_vector in zip(self.kb.documents, self.doc_vectors):
            cosine = sum(w * doc_vector.get(t, 0.0) for t, w in query_vector.items())
            matched_labs = [
                f"{s['test']} {s['status']}" for s in doc["lab_signals"]
                if (s["test"], s["status"]) in wanted
            ]
            score = cosine + LAB_SIGNAL_BONUS * len(matched_labs)
            if score < min_score:
                continue
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "category": doc["category"],
                "score": round(score, 4),
                "matched_terms": sorted(set(query_vector) & set(doc_vector)),
                "matched_lab_signals": matched_labs,
                "content": doc["content"],
                "general_workup": doc["general_workup"],
            })
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]


def build_retrieval_query(symptom_analysis: dict, history_analysis: dict, lab_analysis: dict) -> dict:
    """Turn the analysis agents' outputs into a retrieval query.

    Returns {"query": str, "lab_signals": [{"test", "status"}]}.
    """
    parts: List[str] = list(symptom_analysis.get("key_features", []))
    parts += [c["name"] for c in history_analysis.get("conditions", [])]
    lab_signals = [
        {"test": r["test"], "status": r["status"]}
        for r in lab_analysis.get("results", [])
        if r.get("status") in ("low", "high")
    ]
    parts += [f"{s['test']} {s['status']}" for s in lab_signals]
    return {"query": " ".join(parts), "lab_signals": lab_signals}
