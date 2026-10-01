"""Integration test against a fake OpenAI-compatible server on localhost.

It exercises the real ChatOpenAI client -> base_url -> JSON mode -> Pydantic
parsing path that a local Ollama model (e.g. gemma3:4b) uses - without needing
Ollama or the internet.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app.config import Settings
from app.llm import get_llm
from app.workflow.graph import run_case

DIFFERENTIAL = {"candidates": [{"condition": "Lower respiratory tract infection", "category": "respiratory",
                                "consideration_level": "higher",
                                "supporting_evidence": ["fever", "cough", "WBC high"],
                                "missing_or_conflicting_evidence": [], "knowledge_refs": ["KB-001"]}],
                "reasoning_summary": "A possibility to consider in this simulated case.",
                "disclaimer": "Educational simulation only."}
PLAN = {"steps": [{"step": "Chest X-ray could be considered", "rationale": "Looks for consolidation.",
                   "related_candidates": ["Lower respiratory tract infection"], "priority": "early"}],
        "summary": "Steps to consider.", "disclaimer": "Educational simulation only."}
CASE = {"patient_id": "SIM-001", "age": 45, "sex": "Male", "symptoms": ["fever", "cough", "fatigue"],
        "symptom_duration": "5 days", "lab_results": {"WBC": 13500, "CRP": 48}}


class FakeOpenAIServer:
    """Answers /v1/chat/completions like Ollama's OpenAI-compatible API."""

    def __init__(self, reply_text=None):
        self.requests = []
        self.reply_text = reply_text
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.requests.append(body)
                system = body["messages"][0]["content"]
                content = server.reply_text or json.dumps(
                    DIFFERENTIAL if system.startswith("ROLE: You are the Differential") else PLAN)
                payload = json.dumps({
                    "id": "chatcmpl-test", "object": "chat.completion", "created": 0, "model": body["model"],
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": content}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


@pytest.fixture
def local_settings(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setenv("LLM_STRUCTURED_METHOD", "json_mode")

    def make(server):
        return Settings(openai_model="gemma3:4b", openai_base_url=server.url,
                        structured_output_method="json_mode", request_timeout=10, max_retries=0)
    return make


def test_workflow_with_local_openai_compatible_server(local_settings):
    server = FakeOpenAIServer()
    try:
        settings = local_settings(server)
        state = run_case(CASE, llm=get_llm(settings), settings=settings)
    finally:
        server.httpd.shutdown()

    assert state["differential_diagnosis"]["method"] == "llm"
    assert state["diagnostic_plan"]["method"] == "llm"
    assert state["final_report"]["status"] == "completed"
    assert len(server.requests) == 2
    for request in server.requests:
        assert request["model"] == "gemma3:4b"
        assert request["response_format"] == {"type": "json_object"}
        assert "tools" not in request  # Gemma 3 in Ollama has no tool calling
        assert "RESPONSE FORMAT" in request["messages"][0]["content"]


def test_local_model_returning_bad_json_falls_back(local_settings):
    server = FakeOpenAIServer(reply_text="Sure! Here is the answer: pneumonia")
    try:
        settings = local_settings(server)
        state = run_case(CASE, llm=get_llm(settings), settings=settings)
    finally:
        server.httpd.shutdown()

    assert state["differential_diagnosis"]["method"] == "rule-based"
    assert state["differential_diagnosis"]["candidates"]
    assert state["final_report"]["status"] == "completed"
    assert any("fallback" in e for e in state["errors"])


def test_report_states_actual_reasoning_mode(local_settings):
    server = FakeOpenAIServer()
    try:
        settings = local_settings(server)
        ok = run_case(CASE, llm=get_llm(settings), settings=settings)["final_report"]
    finally:
        server.httpd.shutdown()
    assert ok["llm_mode"] == "llm (gemma3:4b)"

    down = Settings(openai_model="gemma3:4b", openai_base_url="http://127.0.0.1:9/v1",
                    structured_output_method="json_mode", request_timeout=5, max_retries=0)
    report = run_case(CASE, llm=get_llm(down), settings=down)["final_report"]
    assert report["llm_mode"] == "rule-based fallback (gemma3:4b unavailable)"
