"""Small, reusable validation and sanitisation helpers."""

import math
import re
from typing import Any

EDUCATIONAL_DISCLAIMER = (
    "Educational simulation only. This system does not provide medical "
    "diagnosis, treatment, or professional medical advice."
)

# Patterns that look like secrets (OpenAI-style keys, bearer tokens).
_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9_\-\.]{8,}"),
    re.compile(r"(?i)api[_-]?key\s*[=:]\s*\S+"),
]


def normalize_term(term: Any) -> str:
    """Lower-case and trim a free-text clinical term."""
    return " ".join(str(term).strip().lower().split())


def is_valid_number(value: Any) -> bool:
    """True for finite int/float values (bools and NaN/inf are rejected)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def sanitize_error(error: BaseException) -> str:
    """Return a short error description with anything secret-like redacted."""
    message = f"{type(error).__name__}: {error}"
    for pattern in _SECRET_PATTERNS:
        message = pattern.sub("[REDACTED]", message)
    return message[:300]
