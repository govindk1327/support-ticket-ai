"""LLM provider abstraction.

One interface, two HTTP backends. Both Groq and Ollama speak plain JSON over
HTTP, so no vendor SDK is required -- that keeps the dependency list short and
means swapping providers is a config change, not a rewrite.

The client's only job is: take a system prompt and a user message, return the
model's raw text. It knows nothing about QuerySpec, the dataset, or validation.
"""

from __future__ import annotations

import json
from typing import Protocol

import requests

from app.config import Settings, get_settings


class LLMError(Exception):
    """Base class for provider failures."""


class LLMUnavailableError(LLMError):
    """The provider could not be reached, timed out, or rejected credentials.

    Distinct from a bad response: this is the case where falling back to another
    provider (or returning 503) is the right move.
    """


class LLMClient(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


class GroqClient:
    """Groq free tier, OpenAI-compatible chat completions endpoint."""

    name = "groq"

    def __init__(self, settings: Settings | None = None, session=None):
        self.settings = settings or get_settings()
        self._session = session or requests

    def complete(self, system: str, user: str) -> str:
        if not self.settings.groq_api_key:
            raise LLMUnavailableError(
                "GROQ_API_KEY is not set; add it to .env or set LLM_PROVIDER=ollama"
            )

        payload = {
            "model": self.settings.groq_model,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            # Constrains the decoder to emit syntactically valid JSON. It does
            # not guarantee our schema -- Pydantic still does that.
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        try:
            response = self._session.post(
                f"{self.settings.groq_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.settings.groq_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.settings.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise LLMUnavailableError(f"could not reach Groq: {exc}") from exc

        if response.status_code != 200:
            raise LLMUnavailableError(
                f"Groq returned HTTP {response.status_code}: {_safe_body(response)}"
            )

        try:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailableError(f"unexpected Groq response shape: {exc}") from exc


class OllamaClient:
    """Local Ollama. The offline path; no account or key required."""

    name = "ollama"

    def __init__(self, settings: Settings | None = None, session=None):
        self.settings = settings or get_settings()
        self._session = session or requests

    def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.settings.ollama_model,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.settings.temperature,
                "num_predict": self.settings.max_tokens,
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        try:
            response = self._session.post(
                f"{self.settings.ollama_base_url}/api/chat",
                json=payload,
                timeout=self.settings.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise LLMUnavailableError(
                f"could not reach Ollama at {self.settings.ollama_base_url}: {exc}"
            ) from exc

        if response.status_code != 200:
            raise LLMUnavailableError(
                f"Ollama returned HTTP {response.status_code}: {_safe_body(response)}"
            )

        try:
            return response.json()["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMUnavailableError(f"unexpected Ollama response shape: {exc}") from exc


class FallbackClient:
    """Tries the primary provider, then the fallback if it is unreachable.

    Only LLMUnavailableError triggers the fallback. A model that replies with
    nonsense is a different problem, handled by the repair retry -- switching
    providers would not fix it and would double the latency.
    """

    def __init__(self, primary: LLMClient, fallback: LLMClient | None = None):
        self.primary = primary
        self.fallback = fallback
        self.name = primary.name
        self.last_provider: str | None = None
        self.last_error: str | None = None

    def complete(self, system: str, user: str) -> str:
        try:
            result = self.primary.complete(system, user)
            self.last_provider = self.primary.name
            self.last_error = None
            return result
        except LLMUnavailableError as primary_exc:
            self.last_error = str(primary_exc)
            if self.fallback is None:
                raise
            try:
                result = self.fallback.complete(system, user)
                self.last_provider = self.fallback.name
                return result
            except LLMUnavailableError as fallback_exc:
                raise LLMUnavailableError(
                    f"both providers unavailable -- "
                    f"{self.primary.name}: {primary_exc}; "
                    f"{self.fallback.name}: {fallback_exc}"
                ) from fallback_exc


_PROVIDERS = {"groq": GroqClient, "ollama": OllamaClient}


def build_client(settings: Settings | None = None) -> FallbackClient:
    """Construct the configured provider chain."""
    settings = settings or get_settings()

    if settings.provider not in _PROVIDERS:
        raise LLMError(
            f"unknown LLM_PROVIDER '{settings.provider}'; "
            f"valid options are {', '.join(_PROVIDERS)}"
        )

    primary = _PROVIDERS[settings.provider](settings)

    fallback = None
    name = settings.fallback_provider
    if name and name not in ("none", "", settings.provider):
        if name not in _PROVIDERS:
            raise LLMError(f"unknown LLM_FALLBACK_PROVIDER '{name}'")
        fallback = _PROVIDERS[name](settings)

    return FallbackClient(primary, fallback)


def _safe_body(response, limit: int = 200) -> str:
    """Short, non-leaking excerpt of an error body for logs."""
    try:
        body = response.json()
        message = body.get("error", {})
        if isinstance(message, dict):
            message = message.get("message", body)
        return json.dumps(message)[:limit]
    except (ValueError, AttributeError):
        return str(getattr(response, "text", ""))[:limit]
