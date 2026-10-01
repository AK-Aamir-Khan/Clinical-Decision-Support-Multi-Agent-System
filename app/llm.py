"""LLM factory and a helper for structured (Pydantic) LLM calls.

Works with the OpenAI API and with local OpenAI-compatible servers such as
Ollama (e.g. OPENAI_BASE_URL=http://localhost:11434/v1, OPENAI_MODEL=gemma3:4b).
"""

import json
from typing import List, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel

from app.config import Settings, get_settings
from app.utils.validators import sanitize_error

T = TypeVar("T", bound=BaseModel)

JSON_INSTRUCTIONS = """

RESPONSE FORMAT: Reply with ONE JSON object only - no markdown, no code fences, no text before or after it.
Use exactly these keys (the values below only describe the expected type):
{template}"""


def get_llm(settings: Optional[Settings] = None):
    """Return a ChatOpenAI client, or None when no LLM is configured."""
    settings = settings or get_settings()
    if not settings.llm_enabled:
        return None
    from langchain_openai import ChatOpenAI  # imported lazily

    return ChatOpenAI(
        model=settings.openai_model,
        temperature=settings.temperature,
        timeout=settings.request_timeout,
        max_retries=settings.max_retries,
        # local servers ignore the key, but the client requires a value
        api_key=settings.openai_api_key or "not-needed-for-local-server",
        base_url=settings.openai_base_url,
    )


def describe_llm(settings: Optional[Settings] = None) -> str:
    """Human-readable description of the configured LLM (no secrets)."""
    settings = settings or get_settings()
    if not settings.llm_enabled:
        return "No LLM configured - rule-based mode"
    where = f"local server {settings.openai_base_url}" if settings.is_local else "OpenAI API"
    return f"{settings.openai_model} via {where}"


def _example_value(prop: dict, defs: dict):
    """Build a placeholder value from a JSON-schema property."""
    if "$ref" in prop:
        return schema_template(defs[prop["$ref"].split("/")[-1]], defs)
    if "anyOf" in prop:  # e.g. Optional[...]
        return _example_value(next(p for p in prop["anyOf"] if p.get("type") != "null"), defs)
    if "enum" in prop:
        return " | ".join(str(v) for v in prop["enum"])
    kind = prop.get("type")
    if kind == "array":
        return [_example_value(prop.get("items", {}), defs)]
    if kind == "object":
        return schema_template(prop, defs)
    return f"<{kind or 'value'}>"


def schema_template(schema: dict, defs: Optional[dict] = None) -> dict:
    """Turn a Pydantic JSON schema into a compact example object.

    Small local models follow a concrete template more reliably than a raw
    JSON schema (they sometimes echo the schema back instead of filling it).
    """
    defs = defs if defs is not None else schema.get("$defs", {})
    return {name: _example_value(prop, defs) for name, prop in schema.get("properties", {}).items()}


def _with_json_instructions(messages: List[Tuple[str, str]], schema: Type[BaseModel]) -> List[Tuple[str, str]]:
    template = json.dumps(schema_template(schema.model_json_schema()), indent=2)
    role, content = messages[0]
    return [(role, content + JSON_INSTRUCTIONS.format(template=template))] + list(messages[1:])


def invoke_structured(llm, schema: Type[T], messages: List[Tuple[str, str]],
                      method: Optional[str] = None) -> Tuple[Optional[T], Optional[str]]:
    """Call the LLM and parse its answer into `schema`.

    Returns (result, error). Any failure - API error, timeout, malformed or
    schema-invalid output - is returned as a sanitised error string so the
    calling agent can fall back to its deterministic logic.
    """
    if llm is None:
        return None, None
    method = method or get_settings().structured_output_method
    try:
        if method == "json_mode":  # the model is not told the schema, so describe it
            messages = _with_json_instructions(messages, schema)
        structured = llm.with_structured_output(schema, method=method)
        result = structured.invoke(messages)
        if isinstance(result, dict):  # some providers return dicts
            result = schema.model_validate(result)
        if not isinstance(result, schema):
            return None, f"LLM returned unexpected type {type(result).__name__}"
        return result, None
    except Exception as exc:  # noqa: BLE001
        return None, sanitize_error(exc)
