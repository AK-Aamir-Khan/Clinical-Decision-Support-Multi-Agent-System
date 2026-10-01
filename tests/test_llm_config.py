"""LLM configuration: OpenAI API vs. local OpenAI-compatible server (e.g. Ollama)."""

import pytest

from app.config import get_settings
from app.llm import describe_llm, get_llm, invoke_structured
from app.models.schemas import DiagnosticPlanOutput
from tests.fakes import FakeStructuredLLM

ENV_VARS = ["OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "LLM_STRUCTURED_METHOD",
            "LLM_TIMEOUT_SECONDS", "USE_LLM"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV_VARS:  # ignore whatever is in the developer's own .env
        monkeypatch.delenv(name, raising=False)


def test_nothing_configured_means_rule_based():
    settings = get_settings()
    assert settings.llm_enabled is False
    assert get_llm(settings) is None
    assert "rule-based" in describe_llm(settings)


def test_placeholder_key_is_ignored(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "your_api_key_here")
    assert get_settings().llm_enabled is False


def test_openai_defaults(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    settings = get_settings()
    assert settings.llm_enabled and not settings.is_local
    assert settings.structured_output_method == "function_calling"
    assert settings.request_timeout == 30
    assert "test-key-not-real" not in repr(settings)


def test_local_server_needs_no_key(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("OPENAI_MODEL", "gemma3:4b")
    settings = get_settings()

    assert settings.llm_enabled and settings.is_local
    assert settings.structured_output_method == "json_mode"
    assert settings.request_timeout == 180
    assert describe_llm(settings) == "gemma3:4b via local server http://localhost:11434/v1"

    llm = get_llm(settings)
    assert llm.model_name == "gemma3:4b"
    assert llm.openai_api_base == "http://localhost:11434/v1"


def test_method_override_and_invalid_value(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_STRUCTURED_METHOD", "json_schema")
    assert get_settings().structured_output_method == "json_schema"
    monkeypatch.setenv("LLM_STRUCTURED_METHOD", "telepathy")
    assert get_settings().structured_output_method == "json_mode"


PLAN = {"steps": [{"step": "Chest X-ray could be considered", "rationale": "r", "priority": "early"}],
        "summary": "s"}
MESSAGES = [("system", "ROLE: planner"), ("human", "case")]


def test_json_mode_adds_response_template():
    fake = FakeStructuredLLM({"DiagnosticPlanOutput": PLAN})
    result, error = invoke_structured(fake, DiagnosticPlanOutput, MESSAGES, method="json_mode")

    assert error is None and result.steps[0].priority == "early"
    assert fake.methods == ["json_mode"]
    system_prompt = fake.calls[0][1][0][1]
    assert system_prompt.startswith("ROLE: planner")
    assert "RESPONSE FORMAT" in system_prompt
    assert "prompt clinician review | early | routine" in system_prompt


def test_function_calling_leaves_prompt_unchanged():
    fake = FakeStructuredLLM({"DiagnosticPlanOutput": PLAN})
    invoke_structured(fake, DiagnosticPlanOutput, MESSAGES, method="function_calling")
    assert fake.calls[0][1] == MESSAGES
    assert fake.methods == ["function_calling"]
