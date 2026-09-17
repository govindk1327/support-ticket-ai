"""API contract tests.

The LLM is replaced by a dependency override, so these tests exercise routing,
request/response shapes and error mapping with no network and no key. What is
under test is the HTTP layer, not the model.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.data_store import load_data_store
from app.llm.client import LLMUnavailableError
from app.main import app, get_store, get_translator
from app.query.translator import Translator
from tests.stubs import FailingClient, StubClient


@pytest.fixture(scope="module")
def api_store():
    return load_data_store()


@pytest.fixture
def client(api_store):
    """A TestClient whose translator can be swapped per test."""
    app.dependency_overrides[get_store] = lambda: api_store

    def _use(*responses):
        translator = Translator(StubClient(*responses), api_store, get_settings())
        app.dependency_overrides[get_translator] = lambda: translator
        return translator

    with TestClient(app) as test_client:
        test_client.use_llm = _use
        yield test_client

    app.dependency_overrides.clear()


COUNT_OPEN = {
    "intent": "aggregate", "metric": "count",
    "filters": [{"field": "status", "op": "eq", "value": "Open"}],
}


# ==================================================================== /health


def test_health_reports_real_dataset_state(client):
    body = client.get("/health").json()
    assert body["dataset_loaded"] is True
    assert body["row_count"] == 500
    assert body["date_anchor"] == "2024-03-30"
    assert body["date_range"]["min"].startswith("2024-01-01")


def test_health_is_not_hardcoded_ok(client):
    """Status reflects configuration, not a constant string."""
    body = client.get("/health").json()
    assert body["status"] in ("ok", "degraded")
    assert body["status"] == ("ok" if body["llm"]["configured"] else "degraded")


def test_health_reports_llm_configuration(client):
    llm = client.get("/health").json()["llm"]
    assert llm["provider"] in ("groq", "ollama")
    assert llm["model"]
    assert "configured" in llm
    assert llm["reachable"] is None  # not probed by default


def test_health_probe_reports_unreachable_provider(client, api_store):
    translator = Translator(FailingClient(), api_store, get_settings())
    app.dependency_overrides[get_translator] = lambda: translator
    app.state.translator = translator

    body = client.get("/health", params={"probe": True}).json()
    assert body["llm"]["reachable"] is False
    assert body["status"] == "degraded"


def test_health_never_leaks_the_api_key(client):
    body = client.get("/health").text
    assert "GROQ_API_KEY=" not in body
    settings = get_settings()
    if settings.groq_api_key:
        assert settings.groq_api_key not in body


def test_health_surfaces_load_warnings(client):
    body = client.get("/health").json()
    assert any("resolution_time_hrs" in w for w in body["load_warnings"])


# ================================================================ /query (ok)


def test_query_returns_answer_and_result(client):
    client.use_llm(COUNT_OPEN)
    response = client.post("/query", json={"question": "How many tickets are open?"})

    assert response.status_code == 200
    body = response.json()
    assert body["result"]["value"] == 111
    assert "111 tickets" in body["answer"]
    assert body["date_anchor"] == "2024-03-30"
    assert body["attempts"] == 1 and body["repaired"] is False


def test_query_returns_the_interpreted_spec(client):
    """Transparency: the caller can see how the question was read."""
    client.use_llm(COUNT_OPEN)
    spec = client.post("/query", json={"question": "open tickets"}).json()["query_spec"]
    assert spec["intent"] == "aggregate"
    assert spec["filters"][0]["field"] == "status"


def test_query_normalises_values_before_executing(client):
    client.use_llm({"intent": "aggregate", "metric": "count",
                    "filters": [{"field": "category", "op": "eq", "value": "tech"}]})
    body = client.post("/query", json={"question": "tech tickets"}).json()
    assert body["query_spec"]["filters"][0]["value"] == "Technical"
    assert body["result"]["value"] == 152


def test_query_surfaces_null_denominator_warning(client):
    client.use_llm({"intent": "aggregate", "metric": "avg", "target": "customer_rating",
                    "filters": [{"field": "category", "op": "eq", "value": "Technical"}]})
    body = client.post("/query", json={"question": "avg rating for technical"}).json()
    assert body["result"]["value"] == 3.74
    assert any("104 of 152" in w for w in body["warnings"])


def test_query_handles_list_results(client):
    client.use_llm({
        "intent": "list",
        "filters": [{"field": "priority", "op": "eq", "value": "Critical"}],
        "or_groups": [[{"field": "status", "op": "ne", "value": "Resolved"},
                       {"field": "resolution_time_hrs", "op": "gt", "value": 12}]],
    })
    body = client.post("/query", json={"question": "critical not resolved in 12h"}).json()
    assert body["result"]["matched_count"] == 34
    assert len(body["result"]["rows"]) <= 100


def test_query_reports_repair(client):
    client.use_llm(
        {"intent": "aggregate", "metric": "count",
         "filters": [{"field": "priority", "op": "eq", "value": "Urgent"}]},
        {"intent": "aggregate", "metric": "count",
         "filters": [{"field": "priority", "op": "eq", "value": "Critical"}]},
    )
    body = client.post("/query", json={"question": "urgent tickets"}).json()
    assert body["repaired"] is True and body["attempts"] == 2
    assert body["result"]["value"] == 55


def test_query_includes_latency(client):
    client.use_llm(COUNT_OPEN)
    assert client.post("/query", json={"question": "open"}).json()["latency_ms"] >= 0


# =========================================================== /query (invalid)


def test_missing_question_field_is_422(client):
    client.use_llm(COUNT_OPEN)
    assert client.post("/query", json={}).status_code == 422


def test_wrong_type_question_is_422(client):
    client.use_llm(COUNT_OPEN)
    assert client.post("/query", json={"question": {"a": 1}}).status_code == 422


def test_empty_question_is_400(client):
    client.use_llm(COUNT_OPEN)
    response = client.post("/query", json={"question": "   "})
    assert response.status_code == 400
    assert response.json()["stage"] == "input"


def test_oversized_question_is_400(client):
    client.use_llm(COUNT_OPEN)
    response = client.post("/query", json={"question": "x" * 5000})
    assert response.status_code == 400
    assert "too long" in response.json()["message"]


# ==================================================== /query (translation fails)


def test_unfixable_schema_error_is_422(client):
    bad = {"intent": "aggregate", "metric": "count",
           "filters": [{"field": "customer_name", "op": "eq", "value": "Bob"}]}
    client.use_llm(bad, bad)
    response = client.post("/query", json={"question": "Bob's tickets"})

    assert response.status_code == 422
    body = response.json()
    assert body["stage"] == "schema"
    assert body["attempts"] == 2
    assert body["supported_query_types"]


def test_unknown_value_error_returns_valid_values(client):
    bad = {"intent": "aggregate", "metric": "count",
           "filters": [{"field": "status", "op": "eq", "value": "Closed"}]}
    client.use_llm(bad, bad)
    body = client.post("/query", json={"question": "closed tickets"}).json()
    assert body["stage"] == "values"
    assert body["detail"]["valid_values"] == ["Escalated", "Open", "Resolved"]


def test_malformed_json_twice_is_422(client):
    client.use_llm("not json at all", "still not json")
    response = client.post("/query", json={"question": "anything"})
    assert response.status_code == 422
    assert response.json()["stage"] == "parse"


def test_unsupported_question_is_422_with_reason(client):
    client.use_llm({"intent": "unsupported",
                    "reason": "the dataset has no sentiment column"})
    response = client.post("/query", json={"question": "sentiment?"})

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "unsupported_question"
    assert "sentiment" in body["message"]
    assert body["supported_query_types"]


def test_anomaly_question_runs_the_detector(client):
    """The translator interprets; the deterministic detector detects."""
    client.use_llm({"intent": "anomaly"})
    response = client.post("/query", json={"question": "any anomalies?"})

    assert response.status_code == 200
    body = response.json()
    assert body["report"]["total_flagged"] == 286
    assert "286 anomalies" in body["answer"]
    assert body["query_spec"]["intent"] == "anomaly"


def test_anomaly_question_honours_the_time_window(client):
    client.use_llm({"intent": "anomaly", "date_range": {"preset": "this_month"}})
    body = client.post("/query", json={"question": "anomalies this month?"}).json()

    assert body["report"]["total_flagged"] < 286
    assert "this month" in body["answer"]


def test_anomaly_question_makes_no_second_llm_call(client):
    """One call to interpret; the summary sentence is built in Python."""
    translator = client.use_llm({"intent": "anomaly"})
    client.post("/query", json={"question": "anomalies?"})
    assert translator.client.call_count == 1


def test_anomaly_question_applies_priority_filter(client):
    """'Show me anomalies for Critical tickets' must narrow the population."""
    client.use_llm({"intent": "anomaly",
                    "filters": [{"field": "priority", "op": "eq", "value": "Critical"}]})
    body = client.post("/query", json={"question": "anomalies for Critical tickets"}).json()

    assert body["report"]["total_flagged"] == 90
    assert "priority is Critical" in body["report"]["scope"]
    assert body["report"]["matched_summary"]["response_sla_breach"] == 46


def test_anomaly_question_combines_priority_and_date(client):
    client.use_llm({
        "intent": "anomaly",
        "filters": [{"field": "priority", "op": "eq", "value": "High"}],
        "date_range": {"preset": "this_week"},
    })
    body = client.post("/query", json={"question": "high priority anomalies this week"}).json()

    scope = body["report"]["scope"]
    assert "priority is High" in scope and "this week" in scope
    assert body["report"]["total_flagged"] < 286


def test_anomaly_filter_on_a_measured_column_is_rejected(client):
    """Filtering on the column a rule measures would make it circular."""
    client.use_llm({"intent": "anomaly",
                    "filters": [{"field": "resolution_time_hrs", "op": "gt", "value": 50}]})
    response = client.post("/query",
                           json={"question": "anomalies over 50 hours to resolve"})

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "unsupported_anomaly_filter"
    assert body["field"] == "resolution_time_hrs"
    assert "priority" in body["allowed_filter_fields"]


def test_unfiltered_anomaly_question_is_unchanged(client):
    client.use_llm({"intent": "anomaly"})
    body = client.post("/query", json={"question": "any anomalies?"}).json()
    assert body["report"]["total_flagged"] == 286
    assert body["report"]["scope"] is None


# ================================================= /query (provider unavailable)


def test_provider_unavailable_is_503(client, api_store):
    translator = Translator(FailingClient(), api_store, get_settings())
    app.dependency_overrides[get_translator] = lambda: translator

    response = client.post("/query", json={"question": "how many tickets?"})
    assert response.status_code == 503
    assert response.json()["error"] == "llm_unavailable"


def test_both_providers_down_is_503(client, api_store):
    client_stub = StubClient(LLMUnavailableError("groq down; ollama down"))
    translator = Translator(client_stub, api_store, get_settings())
    app.dependency_overrides[get_translator] = lambda: translator

    assert client.post("/query", json={"question": "x"}).status_code == 503


def test_meta_still_works_when_llm_is_down(client, api_store):
    app.dependency_overrides[get_translator] = lambda: Translator(
        FailingClient(), api_store, get_settings())
    assert client.get("/meta").status_code == 200
    assert client.get("/health").status_code == 200


# ====================================================================== /meta


def test_meta_exposes_dataset_metadata(client):
    body = client.get("/meta").json()
    assert body["row_count"] == 500
    assert body["date_anchor"] == "2024-03-30"
    assert body["distinct_values"]["priority"] == ["Critical", "High", "Low", "Medium"]
    assert body["null_counts"]["resolution_time_hrs"] == 173
    assert body["columns"]["created_at"] == "datetime"


def test_meta_matches_the_translation_prompt_context(client, api_store):
    """One source of truth: the UI and the model see the same values."""
    from app.llm.prompts import build_system_prompt

    prompt = build_system_prompt(api_store)
    for value in client.get("/meta").json()["distinct_values"]["status"]:
        assert value in prompt


def test_meta_does_not_expose_secrets_or_paths(client):
    body = client.get("/meta").json()
    assert "api_key" not in str(body).lower()
    assert "groq" not in str(body).lower()


# ================================================================= /anomalies


def test_anomalies_returns_real_flags(client):
    body = client.get("/anomalies", params={"limit": 500}).json()
    assert body["total_flagged"] == 286
    assert body["tickets_flagged"] == 206
    assert body["summary"] == {
        "resolution_time_outlier": 18,
        "unresolved_high_priority": 80,
        "response_sla_breach": 160,
        "data_integrity_contradiction": 28,
    }


def test_anomaly_flags_have_the_structured_shape(client):
    flag = client.get("/anomalies", params={"limit": 1}).json()["flags"][0]
    for field in ("ticket_id", "rule", "severity", "observed_value",
                  "threshold", "explanation"):
        assert field in flag


def test_anomalies_rule_filter(client):
    body = client.get("/anomalies", params={"rule": "data_integrity_contradiction",
                                            "limit": 500}).json()
    assert body["total_flagged"] == 28
    assert {f["rule"] for f in body["flags"]} == {"data_integrity_contradiction"}


def test_anomalies_severity_filter(client):
    body = client.get("/anomalies", params={"severity": "critical", "limit": 500}).json()
    assert body["total_flagged"] == 71
    assert {f["severity"] for f in body["flags"]} == {"critical"}


def test_anomalies_limit_truncates_without_distorting_totals(client):
    body = client.get("/anomalies", params={"limit": 5}).json()
    assert body["returned_count"] == 5
    assert body["total_flagged"] == 286


def test_anomalies_unknown_rule_warns(client):
    body = client.get("/anomalies", params={"rule": "nonsense"}).json()
    assert body["total_flagged"] == 0
    assert any("unknown rule" in w for w in body["warnings"])


def test_anomalies_exposes_thresholds(client):
    thresholds = client.get("/anomalies", params={"limit": 1}).json()["thresholds"]
    assert thresholds["resolution_time_outlier"]["per_priority"]["Critical"]["threshold"] > 0
    assert thresholds["response_sla_breach"]["assumed"] is True


def test_anomalies_reports_known_limitations(client):
    notes = " ".join(client.get("/anomalies", params={"limit": 1}).json()["notes"]).lower()
    assert "snapshot" in notes and "assumed" in notes


def test_anomalies_works_without_an_llm(client, api_store):
    """Detection is fully deterministic, so a dead provider is irrelevant."""
    app.dependency_overrides[get_translator] = lambda: Translator(
        FailingClient(), api_store, get_settings())
    assert client.get("/anomalies", params={"limit": 5}).status_code == 200


def test_anomalies_validates_limit_bounds(client):
    assert client.get("/anomalies", params={"limit": 0}).status_code == 422
    assert client.get("/anomalies", params={"limit": 9999}).status_code == 422


# =================================================================== security


def test_injection_payload_cannot_reach_the_executor(client):
    hostile = {"intent": "aggregate", "metric": "count",
               "filters": [{"field": "__import__('os').system('id')",
                            "op": "eq", "value": "x"}]}
    client.use_llm(hostile, hostile)
    response = client.post("/query", json={"question": "hostile"})
    assert response.status_code == 422
    assert response.json()["stage"] == "schema"


def test_extra_spec_fields_are_rejected(client):
    hostile = {"intent": "aggregate", "metric": "count", "exec": "rm -rf /"}
    client.use_llm(hostile, hostile)
    assert client.post("/query", json={"question": "x"}).status_code == 422


def test_client_cannot_submit_a_raw_query_spec(client):
    """Only a question is accepted; specs come from the translator alone."""
    client.use_llm(COUNT_OPEN)
    response = client.post("/query", json={
        "question": "open tickets",
        "query_spec": {"intent": "list", "limit": 500},
    })
    assert response.status_code == 200
    # The extra field was ignored, not honoured.
    assert response.json()["query_spec"]["intent"] == "aggregate"


def test_error_responses_contain_no_traceback(client):
    client.use_llm("garbage", "garbage")
    body = client.post("/query", json={"question": "x"}).text
    assert "Traceback" not in body
    assert "File \"" not in body


def test_dataset_is_never_mutated_by_requests(client, api_store):
    client.use_llm(COUNT_OPEN)
    client.post("/query", json={"question": "open tickets"})
    assert len(api_store.df) == 500


# ====================================================================== docs


def test_openapi_schema_is_available(client):
    schema = client.get("/openapi.json").json()
    for path in ("/health", "/query", "/anomalies", "/meta"):
        assert path in schema["paths"]
