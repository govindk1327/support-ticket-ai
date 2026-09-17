"""FastAPI application.

Thin HTTP layer over the Part 2/3 pipeline. This module owns routing, request
and response shapes, and error mapping. It owns no query logic: /query hands the
question to the Translator and the resulting spec to the executor, and every
number in the response was computed by pandas.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.anomalies.detector import build_answer, detect
from app.config import AnomalyConfig, Settings, get_settings, load_anomaly_config
from app.data_store import DataStore, SchemaError, load_data_store
from app.llm.client import LLMError, LLMUnavailableError, build_client
from app.models import (
    AnomalyReport,
    Intent,
    QueryResult,
    QuerySpec,
    UnsupportedQueryError,
)
from app.query.executor import execute
from app.query.translator import TranslationError, Translator

logger = logging.getLogger("support_ticket_ai")

API_VERSION = "0.3.0"

# Shown to the user when a question falls outside what the schema can express.
SUPPORTED_SHAPES = [
    "counts and averages (count, avg, sum, min, max, median)",
    "filters on category, priority, status, agent, times and ratings",
    "breakdowns by category, priority, status or agent",
    "time windows such as this month, last week or an explicit date range",
    "listing tickets that match a condition",
]


# --------------------------------------------------------------------------
# Request / response models
# --------------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str = Field(..., description="A natural-language question about the tickets.")


class QueryResponse(BaseModel):
    question: str
    answer: str
    result: QueryResult
    # The interpreted spec is returned deliberately: the user can always see how
    # their question was read, which is the transparency half of the design.
    query_spec: QuerySpec
    date_anchor: str
    provider: str | None = None
    repaired: bool = False
    attempts: int = 1
    warnings: list[str] = Field(default_factory=list)
    latency_ms: int = 0


class AnomalyQueryResponse(BaseModel):
    """Response to an anomaly question asked through /query."""

    question: str
    answer: str
    report: AnomalyReport
    query_spec: QuerySpec
    date_anchor: str
    provider: str | None = None
    repaired: bool = False
    attempts: int = 1
    warnings: list[str] = Field(default_factory=list)
    latency_ms: int = 0


class LLMStatus(BaseModel):
    provider: str
    fallback_provider: str | None = None
    model: str
    configured: bool
    # None unless ?probe=true: a live call costs latency and free-tier quota, so
    # it is opt-in rather than fired on every health check.
    reachable: bool | None = None
    detail: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    dataset_loaded: bool
    row_count: int
    date_range: dict | None = None
    date_anchor: str | None = None
    load_warnings: list[str] = Field(default_factory=list)
    llm: LLMStatus


# --------------------------------------------------------------------------
# Lifespan and dependencies
# --------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the dataset once at startup; 500 rows is cheap to hold in memory."""
    settings = get_settings()
    app.state.settings = settings
    app.state.store = None
    app.state.translator = None
    app.state.startup_error = None
    app.state.anomaly_config = load_anomaly_config()

    try:
        store = load_data_store()
        app.state.store = store
        app.state.translator = Translator(build_client(settings), store, settings)
        logger.info("loaded %s tickets from %s", store.row_count, store.source)
    except (SchemaError, LLMError) as exc:
        # Stay up so /health can explain what is wrong instead of the process
        # dying silently behind a container restart loop.
        app.state.startup_error = str(exc)
        logger.error("startup failed: %s", exc)

    yield


app = FastAPI(
    title="Support Ticket AI",
    version=API_VERSION,
    description="Natural-language querying and anomaly detection over a support ticket dataset.",
    lifespan=lifespan,
)


def get_store(request: Request) -> DataStore:
    store = getattr(request.app.state, "store", None)
    if store is None:
        raise ServiceUnavailable(
            request.app.state.startup_error or "dataset is not loaded"
        )
    return store


def get_translator(request: Request) -> Translator:
    translator = getattr(request.app.state, "translator", None)
    if translator is None:
        raise ServiceUnavailable(
            request.app.state.startup_error or "translator is not available"
        )
    return translator


def get_anomaly_config(request: Request) -> AnomalyConfig:
    config = getattr(request.app.state, "anomaly_config", None)
    return config or load_anomaly_config()


def get_app_settings(request: Request) -> Settings:
    return getattr(request.app.state, "settings", None) or get_settings()


class ServiceUnavailable(Exception):
    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(detail)


# --------------------------------------------------------------------------
# Error handling
#
# Nothing here returns a traceback or echoes configuration. Unexpected errors
# get a request id so a log line can be found without exposing internals.
# --------------------------------------------------------------------------


def _error(status: int, code: str, message: str, **extra) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": code, "message": message, **extra})


@app.exception_handler(ServiceUnavailable)
async def _handle_unavailable(request: Request, exc: ServiceUnavailable):
    return _error(503, "service_unavailable", exc.detail)


@app.exception_handler(TranslationError)
async def _handle_translation(request: Request, exc: TranslationError):
    payload = exc.to_dict()
    # A bad question is the caller's fault (400); a model that could not produce
    # a valid spec is not (422).
    status = 400 if exc.stage == "input" else 422
    if exc.stage != "input":
        payload["supported_query_types"] = SUPPORTED_SHAPES
    return JSONResponse(status_code=status, content=payload)


@app.exception_handler(LLMUnavailableError)
async def _handle_llm_down(request: Request, exc: LLMUnavailableError):
    return _error(
        503,
        "llm_unavailable",
        "No language model provider could be reached. /meta and /health still work.",
        detail=str(exc),
    )


@app.exception_handler(UnsupportedQueryError)
async def _handle_unsupported(request: Request, exc: UnsupportedQueryError):
    payload = exc.to_dict() if hasattr(exc, "to_dict") else {
        "error": "unsupported_query", "message": str(exc)
    }
    payload.setdefault("supported_query_types", SUPPORTED_SHAPES)
    return JSONResponse(status_code=422, content=payload)


@app.exception_handler(Exception)
async def _handle_unexpected(request: Request, exc: Exception):
    request_id = uuid.uuid4().hex[:12]
    logger.exception("unhandled error [%s] on %s", request_id, request.url.path)
    return _error(
        500,
        "internal_error",
        "An unexpected error occurred. Quote the request id when reporting it.",
        request_id=request_id,
    )


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health(request: Request, probe: bool = Query(False, description="Make a live LLM call.")):
    """Real readiness: dataset state plus provider configuration.

    Returns 200 even when degraded, because the point of a health check is to
    say what is wrong, not to refuse to answer.
    """
    settings = get_app_settings(request)
    store: DataStore | None = getattr(request.app.state, "store", None)

    configured = settings.provider != "groq" or bool(settings.groq_api_key)
    model = settings.groq_model if settings.provider == "groq" else settings.ollama_model
    llm = LLMStatus(
        provider=settings.provider,
        fallback_provider=settings.fallback_provider,
        model=model,
        configured=configured,
        detail=None if configured else "GROQ_API_KEY is not set",
    )

    if probe:
        translator: Translator | None = getattr(request.app.state, "translator", None)
        if translator is None:
            llm.reachable = False
            llm.detail = "translator not initialised"
        else:
            try:
                translator.client.complete("Reply with {}.", "ping")
                llm.reachable = True
                llm.detail = None
            except Exception as exc:  # any provider failure is just "not reachable"
                llm.reachable = False
                llm.detail = str(exc)[:200]

    degraded = store is None or not configured or llm.reachable is False
    return HealthResponse(
        status="degraded" if degraded else "ok",
        version=API_VERSION,
        dataset_loaded=store is not None,
        row_count=store.row_count if store else 0,
        date_range=store.metadata()["date_range"] if store else None,
        date_anchor=store.anchor.date().isoformat() if store else None,
        load_warnings=store.load_warnings if store else [],
        llm=llm,
    )


@app.get("/meta", tags=["data"])
def meta(store: DataStore = Depends(get_store)):
    """Dataset metadata: columns, distinct values, nulls, date range.

    Same metadata the translation prompt is built from, so the UI and the model
    always agree about what the dataset contains.
    """
    return store.metadata()


@app.post("/query", tags=["query"])
def query(
    payload: QueryRequest,
    store: DataStore = Depends(get_store),
    translator: Translator = Depends(get_translator),
    config: AnomalyConfig = Depends(get_anomaly_config),
):
    """Answer a natural-language question.

    The LLM translates; pandas computes. Errors are raised as typed exceptions
    and mapped to status codes by the handlers above.
    """
    started = time.perf_counter()

    translation = translator.translate(payload.question)
    spec = translation.spec

    if spec.intent is Intent.ANOMALY:
        # The translator interpreted the question (filters and time window);
        # the deterministic detector does the actual detection. No second LLM
        # call -- the summary sentence is assembled in Python from the counts.
        # Filters that would distort a rule raise AnomalyFilterError, which the
        # handler turns into a 422 rather than silently ignoring them.
        report = detect(store, config=config, spec=spec)

        return AnomalyQueryResponse(
            question=translation.question,
            answer=build_answer(report),
            report=report,
            query_spec=spec,
            date_anchor=store.anchor.date().isoformat(),
            provider=translation.provider,
            repaired=translation.repaired,
            attempts=translation.attempts,
            warnings=report.warnings,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    if spec.intent is Intent.UNSUPPORTED:
        return JSONResponse(
            status_code=422,
            content={
                "error": "unsupported_question",
                "message": spec.reason or "this question cannot be answered from this dataset",
                "supported_query_types": SUPPORTED_SHAPES,
            },
        )

    result = execute(spec, store)

    return QueryResponse(
        question=translation.question,
        answer=result.answer,
        result=result,
        query_spec=spec,
        date_anchor=store.anchor.date().isoformat(),
        provider=translation.provider,
        repaired=translation.repaired,
        attempts=translation.attempts,
        warnings=result.warnings,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


@app.get("/anomalies", response_model=AnomalyReport, tags=["anomalies"])
def anomalies(
    rule: str | None = Query(None, description="Comma-separated rule names."),
    severity: str | None = Query(None, description="Comma-separated severity levels."),
    limit: int = Query(100, ge=1, le=500),
    store: DataStore = Depends(get_store),
    config: AnomalyConfig = Depends(get_anomaly_config),
):
    """Flagged tickets from the four deterministic rules.

    No LLM involvement. Every flag carries the observed value and the threshold
    it crossed, so the finding is checkable rather than asserted.
    """
    return detect(store, config=config, rule=rule, severity=severity, limit=limit)
