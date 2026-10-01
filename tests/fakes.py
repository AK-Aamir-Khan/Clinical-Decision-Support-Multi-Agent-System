"""Fake LLM used in tests - no network or API key required."""


class FakeStructuredLLM:
    """Mimics `llm.with_structured_output(schema).invoke(messages)`.

    `responses` maps a schema class name to an object to return, an
    Exception instance to raise, or a list of these (used in order).
    """

    def __init__(self, responses: dict):
        self.responses = responses
        self.calls = []

    def with_structured_output(self, schema, **kwargs):
        fake = self
        self.methods = getattr(self, "methods", []) + [kwargs.get("method")]

        class _Runnable:
            def invoke(self, messages):
                fake.calls.append((schema.__name__, messages))
                response = fake.responses.get(schema.__name__)
                if isinstance(response, list):
                    response = response.pop(0) if len(response) > 1 else response[0]
                if isinstance(response, Exception):
                    raise response
                if isinstance(response, dict):
                    return schema.model_validate(response)
                return response

        return _Runnable()
