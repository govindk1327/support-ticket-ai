"""Deterministic stand-ins for the LLM, so the suite needs no key and no network.

The translator is tested against scripted model responses -- including the ones
a real model actually gets wrong. Whether a given model produces good specs is a
separate question, measured by the golden set in Part 5 against a live provider.
"""

from __future__ import annotations

import json

from app.llm.client import LLMUnavailableError


class StubClient:
    """Returns scripted responses in order and records what it was asked."""

    def __init__(self, *responses, name: str = "stub"):
        self.responses = list(responses)
        self.name = name
        self.last_provider = name
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self.responses:
            raise AssertionError("StubClient ran out of scripted responses")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, dict):
            return json.dumps(response)
        return response

    @property
    def call_count(self) -> int:
        return len(self.calls)


class FailingClient:
    """A provider that is always unreachable."""

    def __init__(self, name: str = "down", message: str = "connection refused"):
        self.name = name
        self.last_provider = None
        self.message = message
        self.call_count = 0

    def complete(self, system: str, user: str) -> str:
        self.call_count += 1
        raise LLMUnavailableError(f"{self.name}: {self.message}")


class FakeResponse:
    """Minimal stand-in for a requests.Response."""

    def __init__(self, status_code: int = 200, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or json.dumps(payload or {})

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


class FakeSession:
    """Captures the outgoing HTTP call and returns a scripted response."""

    def __init__(self, response=None, exc: Exception | None = None):
        self.response = response
        self.exc = exc
        self.requests: list[dict] = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.requests.append(
            {"url": url, "json": json, "headers": headers, "timeout": timeout}
        )
        if self.exc:
            raise self.exc
        return self.response


def groq_reply(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def ollama_reply(content: str) -> dict:
    return {"message": {"content": content}}
