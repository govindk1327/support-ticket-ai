"""Question -> validated QuerySpec.

The pipeline is deliberately short and closed-ended:

    question -> LLM -> raw text -> JSON -> QuerySpec -> normalised spec
                          |__ one repair retry on failure __|

There is no tool loop and no second model call. The LLM contributes exactly one
thing: an interpretation of what the user meant. Everything downstream is
deterministic Python, and every number the user eventually sees is computed by
pandas from the spec this module returns.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from app.config import Settings, get_settings
from app.data_store import DataStore, normalize_spec
from app.llm.client import LLMUnavailableError
from app.llm.prompts import build_repair_prompt, build_system_prompt
from app.models import QuerySpec, ValueNormalizationError

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class TranslationError(Exception):
    """The question could not be turned into a valid spec.

    Carries structured detail so the API layer can return something actionable
    rather than a bare 500.
    """

    def __init__(self, message: str, *, stage: str, detail: dict | None = None,
                 attempts: int = 0, raw: str | None = None):
        super().__init__(message)
        self.stage = stage  # "input" | "parse" | "schema" | "values"
        self.detail = detail or {}
        self.attempts = attempts
        self.raw = raw

    def to_dict(self) -> dict:
        return {
            "error": "translation_failed",
            "stage": self.stage,
            "message": str(self),
            "attempts": self.attempts,
            **({"detail": self.detail} if self.detail else {}),
        }


@dataclass
class TranslationResult:
    spec: QuerySpec
    question: str
    provider: str | None = None
    attempts: int = 1
    repaired: bool = False
    repair_reason: str | None = None
    raw_responses: list[str] = field(default_factory=list)


def extract_json(text: str) -> dict:
    """Pull a JSON object out of a model response.

    Tolerant of markdown fences and leading prose, because those are the two
    things instruction-tuned models do even when told not to. Tolerance stops
    here: the object itself still has to satisfy the schema.
    """
    if not text or not text.strip():
        raise TranslationError(
            "the model returned an empty response", stage="parse", raw=text
        )

    candidate = text.strip()

    fenced = _FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            raise TranslationError(
                "the model did not return JSON", stage="parse", raw=text
            ) from None
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise TranslationError(
                f"the model returned malformed JSON: {exc}", stage="parse", raw=text
            ) from exc

    if not isinstance(parsed, dict):
        raise TranslationError(
            f"expected a JSON object, got {type(parsed).__name__}",
            stage="parse",
            raw=text,
        )
    return parsed


def _error_details(exc: ValidationError) -> list[dict]:
    """JSON-safe summary of Pydantic errors.

    `exc.errors()` embeds the original exception object under `ctx`, which is
    not serialisable. Only the three fields that are useful to a caller (or to
    the repair prompt) are kept, all coerced to strings.
    """
    return [
        {
            "field": ".".join(str(p) for p in err["loc"]) or "spec",
            "message": str(err["msg"]),
            "type": str(err["type"]),
        }
        for err in exc.errors()[:5]
    ]


def _format_validation_error(exc: ValidationError) -> str:
    """Compact, model-readable summary of what Pydantic rejected."""
    return "; ".join(f"{d['field']}: {d['message']}" for d in _error_details(exc))


class Translator:
    """Turns a natural-language question into a validated, normalised QuerySpec."""

    def __init__(
        self,
        client,
        store: DataStore,
        settings: Settings | None = None,
    ):
        self.client = client
        self.store = store
        self.settings = settings or get_settings()
        self.system_prompt = build_system_prompt(store)

    # -- internals ---------------------------------------------------------

    def _to_spec(self, raw: str) -> QuerySpec:
        """Parse, validate, normalise. Raises TranslationError on any failure."""
        payload = extract_json(raw)

        try:
            spec = QuerySpec.model_validate(payload)
        except ValidationError as exc:
            raise TranslationError(
                _format_validation_error(exc),
                stage="schema",
                detail={"errors": _error_details(exc)},
                raw=raw,
            ) from exc

        try:
            return normalize_spec(spec, self.store)
        except ValueNormalizationError as exc:
            raise TranslationError(
                str(exc), stage="values", detail=exc.to_dict(), raw=raw
            ) from exc

    def _check_question(self, question: str) -> str:
        cleaned = (question or "").strip()
        if not cleaned:
            raise TranslationError("the question is empty", stage="input")
        if len(cleaned) > self.settings.max_question_length:
            raise TranslationError(
                f"the question is too long "
                f"({len(cleaned)} characters, limit {self.settings.max_question_length})",
                stage="input",
            )
        return cleaned

    # -- public API --------------------------------------------------------

    def translate(self, question: str) -> TranslationResult:
        """Translate a question, with at most one repair attempt.

        Raises TranslationError if the spec is still invalid after the retry,
        and LLMUnavailableError if no provider could be reached.
        """
        question = self._check_question(question)

        raw = self.client.complete(self.system_prompt, question)
        responses = [raw]

        try:
            spec = self._to_spec(raw)
            return TranslationResult(
                spec=spec,
                question=question,
                provider=getattr(self.client, "last_provider", None),
                attempts=1,
                raw_responses=responses,
            )
        except TranslationError as exc:
            if self.settings.max_repair_attempts < 1:
                exc.attempts = 1
                raise
            # Python unbinds the `as` name at the end of the except block, so
            # hold a reference for the repair prompt below.
            first_error = exc

        # One retry, with the specific failure handed back. Capped on purpose:
        # a model that cannot fix a named error in one turn is unlikely to fix
        # it in three, and each turn costs a full round trip.
        repair_message = build_repair_prompt(question, raw, str(first_error))
        try:
            retry_raw = self.client.complete(self.system_prompt, repair_message)
        except LLMUnavailableError:
            raise
        responses.append(retry_raw)

        try:
            spec = self._to_spec(retry_raw)
        except TranslationError as second_error:
            second_error.attempts = 2
            raise second_error from None

        return TranslationResult(
            spec=spec,
            question=question,
            provider=getattr(self.client, "last_provider", None),
            attempts=2,
            repaired=True,
            repair_reason=str(first_error),
            raw_responses=responses,
        )
