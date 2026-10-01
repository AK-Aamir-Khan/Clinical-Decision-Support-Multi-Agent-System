"""Loads the local, educational knowledge base (data/knowledge_base.json)."""

import json
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_KB_PATH = Path(__file__).resolve().parents[2] / "data" / "knowledge_base.json"

REQUIRED_FIELDS = ("id", "title", "keywords", "content", "general_workup")


class KnowledgeBase:
    """In-memory collection of educational documents."""

    def __init__(self, path: Optional[Path] = None):
        path = Path(path) if path else DEFAULT_KB_PATH
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        self.meta: dict = raw.get("_meta", {})
        self.documents: List[dict] = []
        for doc in raw.get("documents", []):
            missing = [f for f in REQUIRED_FIELDS if f not in doc]
            if missing:
                raise ValueError(f"Knowledge base document {doc.get('id')} missing {missing}")
            doc.setdefault("lab_signals", [])
            doc.setdefault("category", "general")
            self.documents.append(doc)
        self._by_id: Dict[str, dict] = {d["id"]: d for d in self.documents}

    def get(self, doc_id: str) -> Optional[dict]:
        return self._by_id.get(doc_id)

    def __len__(self) -> int:
        return len(self.documents)

    @staticmethod
    def searchable_text(doc: dict) -> str:
        """Text indexed by the retriever: title, keywords, lab signals, content."""
        signals = " ".join(f"{s['test']} {s['status']}" for s in doc.get("lab_signals", []))
        return " ".join([doc["title"], " ".join(doc["keywords"]), signals, doc["content"]])
