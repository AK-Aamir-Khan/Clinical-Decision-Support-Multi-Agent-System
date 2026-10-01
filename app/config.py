"""Central configuration, read from environment variables (.env supported).

Secrets are never hard-coded. The LLM can be the OpenAI API (OPENAI_API_KEY)
or a local OpenAI-compatible server such as Ollama (OPENAI_BASE_URL). If
neither is configured, the system runs in deterministic "rule-based" mode so
it can still be demonstrated and tested.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

load_dotenv(PROJECT_ROOT / ".env")


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


STRUCTURED_METHODS = ("function_calling", "json_mode", "json_schema")


@dataclass(frozen=True)
class Settings:
    openai_api_key: Optional[str] = field(default=None, repr=False)  # never printed
    openai_model: str = "gpt-4o-mini"
    # Any OpenAI-compatible server, e.g. Ollama: http://localhost:11434/v1
    openai_base_url: Optional[str] = None
    # How structured output is requested. Local models such as Gemma 3 do not
    # support tool calling, so "json_mode" is the default when a base URL is set.
    structured_output_method: str = "function_calling"
    temperature: float = 0.0
    request_timeout: int = 30
    max_retries: int = 1
    use_llm: bool = True
    rag_top_k: int = 4
    max_safety_revisions: int = 1

    @property
    def llm_enabled(self) -> bool:
        """An OpenAI key OR a local OpenAI-compatible server enables the LLM."""
        return self.use_llm and bool(self.openai_api_key or self.openai_base_url)

    @property
    def is_local(self) -> bool:
        return bool(self.openai_base_url)


def get_settings() -> Settings:
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if key in ("", "your_api_key_here"):
        key = None
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip() or None

    default_method = "json_mode" if base_url else "function_calling"
    method = (os.getenv("LLM_STRUCTURED_METHOD") or default_method).strip().lower()
    if method not in STRUCTURED_METHODS:
        method = default_method

    return Settings(
        openai_api_key=key,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        openai_base_url=base_url,
        structured_output_method=method,
        temperature=_float_env("LLM_TEMPERATURE", 0.0),
        # small local models on a laptop CPU are much slower than the OpenAI API
        request_timeout=_int_env("LLM_TIMEOUT_SECONDS", 180 if base_url else 30),
        max_retries=_int_env("LLM_MAX_RETRIES", 1),
        use_llm=os.getenv("USE_LLM", "true").strip().lower() not in ("0", "false", "no"),
        rag_top_k=_int_env("RAG_TOP_K", 4),
        max_safety_revisions=_int_env("MAX_SAFETY_REVISIONS", 1),
    )
