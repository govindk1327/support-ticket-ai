"""QuerySpec validation.

This is the boundary between the LLM and the data. Everything the model can
plausibly get wrong should be rejected here, before the executor runs, with a
message specific enough to feed back into a repair retry (Part 3).
"""

import pytest
from pydantic import ValidationError

from app.models import (
    DatePreset,
    Filter,
    Intent,
    Metric,
    Op,
    QuerySpec,
    Sort,
)


def spec_from(payload: dict) -> QuerySpec:
    """Mirrors how Part 3 will construct a spec from parsed model JSON."""
    return QuerySpec.model_validate(payload)


# ------------------------------------------------------- hallucinated names


def test_rejects_hallucinated_column():
    with pytest.raises(ValidationError, match="unknown column 'sentiment_score'"):
        Filter(field="sentiment_score", op=Op.EQ, value="positive")


def test_error_lists_valid_columns_for_repair():
    with pytest.raises(ValidationError) as exc:
        Filter(field="agent_name", op=Op.EQ, value="x")
    assert "agent_id" in str(exc.value)


def test_rejects_invented_operator():
    with pytest.raises(ValidationError):
        spec_from({"intent": "list", "filters": [{"field": "status", "op": "like", "value": "Open"}]})


def test_rejects_invented_intent():
    with pytest.raises(ValidationError):
        spec_from({"intent": "summarise"})


def test_rejects_unknown_extra_fields():
    """A model inventing its own field must fail rather than have it ignored."""
    with pytest.raises(ValidationError):
        spec_from({"intent": "aggregate", "metric": "count", "sql": "SELECT * FROM t"})


# --------------------------------------------------- type / op coherence


def test_rejects_numeric_operator_on_text_column():
    with pytest.raises(ValidationError, match="only applies to numeric"):
        Filter(field="category", op=Op.GT, value=5)


def test_rejects_string_value_on_numeric_column():
    with pytest.raises(ValidationError, match="must be a number"):
        Filter(field="resolution_time_hrs", op=Op.GT, value="twelve")


def test_rejects_numeric_value_on_text_column():
    with pytest.raises(ValidationError, match="must be a string"):
        Filter(field="status", op=Op.EQ, value=3)


def test_rejects_contains_on_numeric_column():
    with pytest.raises(ValidationError, match="only applies to text"):
        Filter(field="customer_rating", op=Op.CONTAINS, value="5")


def test_in_requires_non_empty_list():
    with pytest.raises(ValidationError, match="non-empty list"):
        Filter(field="priority", op=Op.IN, value=[])
    with pytest.raises(ValidationError, match="non-empty list"):
        Filter(field="priority", op=Op.IN, value="Critical")


def test_scalar_op_rejects_list_value():
    with pytest.raises(ValidationError, match="not a list"):
        Filter(field="priority", op=Op.EQ, value=["Critical", "High"])


def test_null_ops_take_no_value():
    with pytest.raises(ValidationError, match="takes no value"):
        Filter(field="customer_rating", op=Op.IS_NULL, value=3)
    assert Filter(field="customer_rating", op=Op.IS_NULL).value is None


def test_missing_value_is_rejected():
    with pytest.raises(ValidationError, match="requires a value"):
        Filter(field="status", op=Op.EQ)


# ------------------------------------------------------- spec coherence


def test_aggregate_requires_metric():
    with pytest.raises(ValidationError, match="require a metric"):
        spec_from({"intent": "aggregate"})


def test_avg_requires_numeric_target():
    with pytest.raises(ValidationError, match="needs a numeric target"):
        spec_from({"intent": "aggregate", "metric": "avg", "target": "category"})


def test_avg_without_target_is_rejected():
    with pytest.raises(ValidationError, match="requires a target"):
        spec_from({"intent": "aggregate", "metric": "avg"})


def test_count_drops_a_stray_target_instead_of_failing():
    """Models often attach a target to count. Harmless; normalise it away."""
    spec = spec_from({"intent": "aggregate", "metric": "count", "target": "ticket_id"})
    assert spec.target is None


def test_list_intent_rejects_metric():
    with pytest.raises(ValidationError, match="does not take a metric"):
        spec_from({"intent": "list", "metric": "count"})


def test_group_by_must_be_categorical():
    with pytest.raises(ValidationError, match="cannot group by"):
        spec_from({"intent": "aggregate", "metric": "count", "group_by": "resolution_time_hrs"})
    with pytest.raises(ValidationError, match="cannot group by"):
        spec_from({"intent": "aggregate", "metric": "count", "group_by": "ticket_id"})


def test_group_by_rejected_on_list_intent():
    with pytest.raises(ValidationError, match="only valid for aggregate"):
        spec_from({"intent": "list", "group_by": "agent_id"})


def test_sort_by_value_requires_group_by():
    with pytest.raises(ValidationError, match="only valid with group_by"):
        spec_from({"intent": "aggregate", "metric": "count", "sort": {"field": "value"}})


def test_sort_rejects_unknown_field():
    with pytest.raises(ValidationError, match="cannot sort by"):
        Sort(field="relevance")


def test_or_group_needs_two_branches():
    with pytest.raises(ValidationError, match="at least two filters"):
        spec_from({
            "intent": "list",
            "or_groups": [[{"field": "status", "op": "eq", "value": "Open"}]],
        })


def test_unsupported_requires_a_reason():
    with pytest.raises(ValidationError, match="explain why"):
        spec_from({"intent": "unsupported"})
    assert spec_from({"intent": "unsupported", "reason": "no such column"}).reason


def test_limit_bounds_are_enforced():
    with pytest.raises(ValidationError):
        spec_from({"intent": "list", "limit": 0})
    with pytest.raises(ValidationError):
        spec_from({"intent": "list", "limit": -5})
    with pytest.raises(ValidationError):
        spec_from({"intent": "list", "limit": 100000})


# ------------------------------------------------------------ valid specs


def test_minimal_valid_specs_parse():
    assert spec_from({"intent": "aggregate", "metric": "count"}).metric is Metric.COUNT
    assert spec_from({"intent": "list"}).filters == []
    assert spec_from({"intent": "anomaly"}).intent is Intent.ANOMALY


def test_full_spec_round_trips():
    payload = {
        "intent": "aggregate",
        "metric": "avg",
        "target": "customer_rating",
        "filters": [{"field": "category", "op": "eq", "value": "Technical"}],
        "or_groups": [[
            {"field": "status", "op": "ne", "value": "Resolved"},
            {"field": "resolution_time_hrs", "op": "gt", "value": 12},
        ]],
        "group_by": "agent_id",
        "date_range": {"preset": "this_month"},
        "sort": {"field": "value", "direction": "asc"},
        "limit": 5,
    }
    spec = spec_from(payload)
    assert spec.date_range.preset is DatePreset.THIS_MONTH
    assert len(spec.or_groups[0]) == 2
    assert spec_from(spec.model_dump(mode="json", exclude_none=True)) == spec


def test_date_range_preset_must_be_known():
    with pytest.raises(ValidationError):
        spec_from({"intent": "list", "date_range": {"preset": "last_fortnight"}})


# ------------------------------------------------------ date-column value types


def test_date_column_requires_a_string_value():
    with pytest.raises(ValidationError, match="must be a 'YYYY-MM-DD' string"):
        Filter(field="created_at", op=Op.EQ, value=20240305)


def test_date_column_list_values_must_be_strings():
    with pytest.raises(ValidationError, match="YYYY-MM-DD"):
        Filter(field="created_at", op=Op.IN, value=["2024-03-05", 7])


def test_ordering_op_on_date_column_still_directs_to_date_range():
    with pytest.raises(ValidationError, match="date_range"):
        Filter(field="created_at", op=Op.GT, value="2024-03-05")


# ---------------------------------------------------- grouped sort constraints


def test_grouped_sort_must_target_value_or_group_key():
    with pytest.raises(ValidationError, match="can only be sorted by"):
        QuerySpec(
            intent=Intent.AGGREGATE,
            metric=Metric.COUNT,
            group_by="agent_id",
            sort=Sort(field="resolution_time_hrs", direction="desc"),
        )


def test_grouped_sort_accepts_value_and_group_key():
    for field in ("value", "agent_id"):
        spec = QuerySpec(
            intent=Intent.AGGREGATE,
            metric=Metric.COUNT,
            group_by="agent_id",
            sort=Sort(field=field, direction="asc"),
        )
        assert spec.sort.field == field
