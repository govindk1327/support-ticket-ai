"""LLM provider clients and the fallback chain.

All HTTP is mocked. These tests check the request we send, the response we
accept, and -- most importantly -- that unreachable is distinguished from
wrong, because only the former should trigger a fallback.
"""

from __future__ import annotations

import dataclasses

import pytest
import requests

from app.config import Settings
from app.llm.client import (
    FallbackClient,
    GroqClient,
    LLMError,
    LLMUnavailableError,
    OllamaClient,
    build_client,
)
from tests.stubs import (
    FailingClient,
    FakeResponse,
    FakeSession,
    StubClient,
    groq_reply,
    ollama_reply,
)


@pytest.fixture
def settings():
    return Settings(
        provider="groq",
        fallback_provider="ollama",
        groq_api_key="test-key",
        groq_model="test-model",
    )


# ==================================================================== Groq


def test_groq_sends_a_well_formed_request(settings):
    session = FakeSession(FakeResponse(200, groq_reply('{"intent":"aggregate"}')))
    client = GroqClient(settings, session=session)

    assert client.complete("SYSTEM", "USER") == '{"intent":"aggregate"}'

    sent = session.requests[0]
    assert sent["url"].endswith("/chat/completions")
    assert sent["headers"]["Authorization"] == "Bearer test-key"
    body = sent["json"]
    assert body["model"] == "test-model"
    assert body["temperature"] == 0.0
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"][0] == {"role": "system", "content": "SYSTEM"}
    assert body["messages"][1] == {"role": "user", "content": "USER"}


def test_groq_requires_an_api_key(settings):
    keyless = dataclasses.replace(settings, groq_api_key=None)
    client = GroqClient(keyless, session=FakeSession())
    with pytest.raises(LLMUnavailableError, match="GROQ_API_KEY"):
        client.complete("s", "u")


@pytest.mark.parametrize("status", [401, 429, 500, 503])
def test_groq_http_errors_are_unavailable(settings, status):
    session = FakeSession(FakeResponse(status, {"error": {"message": "nope"}}))
    client = GroqClient(settings, session=session)
    with pytest.raises(LLMUnavailableError) as exc:
        client.complete("s", "u")
    assert str(status) in str(exc.value)


def test_groq_network_error_is_unavailable(settings):
    session = FakeSession(exc=requests.ConnectionError("dns failure"))
    client = GroqClient(settings, session=session)
    with pytest.raises(LLMUnavailableError, match="could not reach Groq"):
        client.complete("s", "u")


def test_groq_timeout_is_unavailable(settings):
    session = FakeSession(exc=requests.Timeout("timed out"))
    client = GroqClient(settings, session=session)
    with pytest.raises(LLMUnavailableError):
        client.complete("s", "u")


def test_groq_unexpected_body_is_unavailable(settings):
    session = FakeSession(FakeResponse(200, {"unexpected": "shape"}))
    client = GroqClient(settings, session=session)
    with pytest.raises(LLMUnavailableError, match="unexpected Groq response"):
        client.complete("s", "u")


def test_groq_error_body_does_not_leak_the_key(settings):
    session = FakeSession(FakeResponse(401, {"error": {"message": "bad key"}}))
    client = GroqClient(settings, session=session)
    with pytest.raises(LLMUnavailableError) as exc:
        client.complete("s", "u")
    assert "test-key" not in str(exc.value)


# ================================================================== Ollama


def test_ollama_sends_a_well_formed_request(settings):
    session = FakeSession(FakeResponse(200, ollama_reply('{"intent":"list"}')))
    client = OllamaClient(settings, session=session)

    assert client.complete("SYSTEM", "USER") == '{"intent":"list"}'

    body = session.requests[0]["json"]
    assert session.requests[0]["url"].endswith("/api/chat")
    assert body["format"] == "json"
    assert body["stream"] is False
    assert body["options"]["temperature"] == 0.0


def test_ollama_needs_no_api_key(settings):
    keyless = dataclasses.replace(settings, groq_api_key=None)
    session = FakeSession(FakeResponse(200, ollama_reply("{}")))
    assert OllamaClient(keyless, session=session).complete("s", "u") == "{}"


def test_ollama_connection_error_names_the_url(settings):
    session = FakeSession(exc=requests.ConnectionError("refused"))
    client = OllamaClient(settings, session=session)
    with pytest.raises(LLMUnavailableError, match="localhost:11434"):
        client.complete("s", "u")


def test_ollama_http_error_is_unavailable(settings):
    session = FakeSession(FakeResponse(404, {"error": "model not found"}))
    client = OllamaClient(settings, session=session)
    with pytest.raises(LLMUnavailableError):
        client.complete("s", "u")


# ================================================================ fallback


def test_primary_is_used_when_healthy():
    primary = StubClient("ok", name="groq")
    fallback = StubClient("unused", name="ollama")
    chain = FallbackClient(primary, fallback)

    assert chain.complete("s", "u") == "ok"
    assert chain.last_provider == "groq"
    assert fallback.call_count == 0


def test_fallback_is_used_when_primary_is_down():
    primary = FailingClient(name="groq")
    fallback = StubClient("rescued", name="ollama")
    chain = FallbackClient(primary, fallback)

    assert chain.complete("s", "u") == "rescued"
    assert chain.last_provider == "ollama"
    assert chain.last_error is not None


def test_both_providers_down_reports_both():
    chain = FallbackClient(FailingClient("groq"), FailingClient("ollama"))
    with pytest.raises(LLMUnavailableError) as exc:
        chain.complete("s", "u")
    message = str(exc.value)
    assert "groq" in message and "ollama" in message


def test_no_fallback_configured_propagates():
    chain = FallbackClient(FailingClient("groq"), None)
    with pytest.raises(LLMUnavailableError):
        chain.complete("s", "u")


def test_bad_model_output_does_not_trigger_fallback():
    """Switching providers cannot fix nonsense; the repair retry handles that."""
    primary = StubClient("this is not json", name="groq")
    fallback = StubClient("unused", name="ollama")
    chain = FallbackClient(primary, fallback)

    assert chain.complete("s", "u") == "this is not json"
    assert fallback.call_count == 0


def test_fallback_recovers_then_reports_correct_provider():
    chain = FallbackClient(FailingClient("groq"), StubClient("{}", name="ollama"))
    chain.complete("s", "u")
    assert chain.last_provider == "ollama"


# ================================================================== factory


def test_build_client_defaults_to_groq_with_ollama_fallback(settings):
    chain = build_client(settings)
    assert isinstance(chain.primary, GroqClient)
    assert isinstance(chain.fallback, OllamaClient)
    assert chain.name == "groq"


def test_build_client_honours_provider_choice(settings):
    chain = build_client(dataclasses.replace(settings, provider="ollama",
                                             fallback_provider="none"))
    assert isinstance(chain.primary, OllamaClient)
    assert chain.fallback is None


def test_build_client_skips_self_fallback(settings):
    chain = build_client(dataclasses.replace(settings, provider="groq",
                                             fallback_provider="groq"))
    assert chain.fallback is None


def test_build_client_rejects_unknown_provider(settings):
    with pytest.raises(LLMError, match="unknown LLM_PROVIDER"):
        build_client(dataclasses.replace(settings, provider="openai"))


def test_build_client_rejects_unknown_fallback(settings):
    with pytest.raises(LLMError, match="unknown LLM_FALLBACK_PROVIDER"):
        build_client(dataclasses.replace(settings, fallback_provider="openai"))


def test_swapping_providers_is_a_config_change_only(settings):
    """The interface is identical, which is what makes the swap a one-liner."""
    for provider in ("groq", "ollama"):
        chain = build_client(dataclasses.replace(settings, provider=provider))
        assert hasattr(chain, "complete")


def test_translation_settings_are_deterministic(settings):
    assert settings.temperature == 0.0
    assert settings.max_repair_attempts == 1
