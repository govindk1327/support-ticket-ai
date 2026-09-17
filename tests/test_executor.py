"""Executor tests against the real dataset.

Expected values were computed independently from the shipped CSV. This is the
layer that produces every number a user sees, so it carries the bulk of the
test weight.
"""

import math

import pytest

from app.models import (
    DatePreset,
    DateRange,
    Filter,
    Intent,
    Metric,
    Op,
    QuerySpec,
    Sort,
    UnsupportedQueryError,
)
from app.query.executor import execute


def agg(**kw):
    return QuerySpec(intent=Intent.AGGREGATE, **kw)


def lst(**kw):
    return QuerySpec(intent=Intent.LIST, **kw)


# ---------------------------------------------------------------- counting


def test_count_all_tickets(store):
    r = execute(agg(metric=Metric.COUNT), store)
    assert r.value == 500
    assert r.answer == "500 tickets."


def test_count_open_tickets(store):
    """Sample query 1: 'How many tickets are currently open?'"""
    r = execute(
        agg(metric=Metric.COUNT, filters=[Filter(field="status", op=Op.EQ, value="Open")]),
        store,
    )
    assert r.value == 111
    assert "111 tickets" in r.answer
    assert "status is Open" in r.answer


def test_count_is_case_insensitive_on_text(store):
    a = execute(agg(metric=Metric.COUNT, filters=[Filter(field="status", op=Op.EQ, value="open")]), store)
    b = execute(agg(metric=Metric.COUNT, filters=[Filter(field="status", op=Op.EQ, value="OPEN")]), store)
    assert a.value == b.value == 111


def test_count_with_two_and_filters(store):
    r = execute(
        agg(
            metric=Metric.COUNT,
            filters=[
                Filter(field="category", op=Op.EQ, value="Billing"),
                Filter(field="priority", op=Op.EQ, value="High"),
            ],
        ),
        store,
    )
    assert r.value == 45


def test_count_with_date_preset(store):
    r = execute(agg(metric=Metric.COUNT, date_range=DateRange(preset=DatePreset.THIS_MONTH)), store)
    assert r.value == 188
    assert r.date_range.start == "2024-03-01"
    assert r.date_range.end == "2024-03-31"


# ------------------------------------------------------------- aggregation


def test_avg_rating_for_technical_reports_denominator(store):
    """Sample query 4. The null denominator is the point of this test."""
    r = execute(
        agg(
            metric=Metric.AVG,
            target="customer_rating",
            filters=[Filter(field="category", op=Op.EQ, value="Technical")],
        ),
        store,
    )
    assert r.value == 3.74
    assert r.n_used == 104
    assert r.n_total == 152
    assert any("104 of 152" in w for w in r.warnings)


def test_count_metric_has_no_null_warning(store):
    r = execute(agg(metric=Metric.COUNT), store)
    assert r.warnings == []


def test_median_and_max_resolution_time(store):
    assert execute(agg(metric=Metric.MEDIAN, target="resolution_time_hrs"), store).value == 12.0
    assert execute(agg(metric=Metric.MAX, target="resolution_time_hrs"), store).value == 119.7


def test_aggregate_over_all_null_slice_returns_none(store):
    """Open tickets have no rating at all; report that, don't emit NaN."""
    r = execute(
        agg(
            metric=Metric.AVG,
            target="customer_rating",
            filters=[Filter(field="status", op=Op.EQ, value="Open")],
        ),
        store,
    )
    assert r.value is None
    assert r.n_used == 0
    assert r.n_total == 111
    assert "No customer rating values available" in r.answer


# ------------------------------------------------------------------ groups


def test_group_by_agent_resolved_this_month(store):
    """Sample query 2: 'Which agent resolved the most tickets this month?'"""
    r = execute(
        agg(
            metric=Metric.COUNT,
            filters=[Filter(field="status", op=Op.EQ, value="Resolved")],
            date_range=DateRange(preset=DatePreset.THIS_MONTH),
            group_by="agent_id",
            sort=Sort(field="value", direction="desc"),
        ),
        store,
    )
    assert len(r.groups) == 12
    assert r.groups[0].group == "AGT-01"
    assert r.groups[0].value == 16
    assert "AGT-01" in r.answer


def test_group_by_defaults_to_descending_value(store):
    """A spec with no sort must still put the largest group first."""
    r = execute(agg(metric=Metric.COUNT, group_by="status"), store)
    assert [g.group for g in r.groups] == ["Resolved", "Open", "Escalated"]
    assert [g.value for g in r.groups] == [327, 111, 62]


def test_group_by_ascending_finds_lowest_average(store):
    """'Which agent has the lowest average customer rating?' (brief, section 2)"""
    r = execute(
        agg(
            metric=Metric.AVG,
            target="customer_rating",
            group_by="agent_id",
            sort=Sort(field="value", direction="asc"),
            limit=1,
        ),
        store,
    )
    assert r.groups[0].group == "AGT-08"
    assert r.groups[0].value == 3.48
    assert r.groups[0].n_used == 25
    assert "lowest" in r.answer


def test_group_limit_truncates_after_sorting(store):
    r = execute(
        agg(metric=Metric.COUNT, group_by="agent_id", sort=Sort(field="value"), limit=3),
        store,
    )
    assert len(r.groups) == 3
    assert [g.value for g in r.groups] == sorted([g.value for g in r.groups], reverse=True)


def test_grouping_empty_result_is_clean(store):
    r = execute(
        agg(
            metric=Metric.COUNT,
            group_by="agent_id",
            filters=[Filter(field="category", op=Op.EQ, value="Billing"),
                     Filter(field="issue_summary", op=Op.CONTAINS, value="zzzz")],
        ),
        store,
    )
    assert r.groups == []
    assert "nothing to group" in r.answer


# --------------------------------------------------------------- or_groups


def test_or_group_handles_critical_sample_query(store):
    """Sample query 3: 'All Critical tickets not resolved within 12 hours.'

    Correct reading is Critical AND (not resolved OR took >12h). An AND-only
    spec gets 31 or 3 depending on which clause it picks; the right answer is 34.
    """
    r = execute(
        lst(
            filters=[Filter(field="priority", op=Op.EQ, value="Critical")],
            or_groups=[[
                Filter(field="status", op=Op.NE, value="Resolved"),
                Filter(field="resolution_time_hrs", op=Op.GT, value=12),
            ]],
            limit=100,
        ),
        store,
    )
    assert r.matched_count == 34


def test_or_group_is_strictly_wider_than_either_clause(store):
    base = [Filter(field="priority", op=Op.EQ, value="Critical")]
    unresolved = execute(
        agg(metric=Metric.COUNT, filters=base + [Filter(field="status", op=Op.NE, value="Resolved")]),
        store,
    ).value
    slow = execute(
        agg(metric=Metric.COUNT, filters=base + [Filter(field="resolution_time_hrs", op=Op.GT, value=12)]),
        store,
    ).value
    assert (unresolved, slow) == (31, 3)


def test_or_group_appears_in_answer_text(store):
    r = execute(
        agg(
            metric=Metric.COUNT,
            filters=[Filter(field="priority", op=Op.EQ, value="Critical")],
            or_groups=[[
                Filter(field="status", op=Op.NE, value="Resolved"),
                Filter(field="resolution_time_hrs", op=Op.GT, value=12),
            ]],
        ),
        store,
    )
    assert " or " in r.answer


# --------------------------------------------------------------- operators


@pytest.mark.parametrize(
    "op,value,expected",
    [
        (Op.IN, ["Critical", "High"], 189),
        (Op.NOT_IN, ["Critical", "High"], 311),
        (Op.NE, "Resolved", 173),
    ],
)
def test_membership_operators(store, op, value, expected):
    field = "priority" if op in (Op.IN, Op.NOT_IN) else "status"
    r = execute(agg(metric=Metric.COUNT, filters=[Filter(field=field, op=op, value=value)]), store)
    assert r.value == expected


def test_contains_is_case_insensitive_substring(store):
    r = execute(
        agg(metric=Metric.COUNT, filters=[Filter(field="issue_summary", op=Op.CONTAINS, value="INVOICE")]),
        store,
    )
    assert r.value == 32


def test_null_operators_partition_the_dataset(store):
    is_null = execute(
        agg(metric=Metric.COUNT, filters=[Filter(field="resolution_time_hrs", op=Op.IS_NULL)]), store
    ).value
    not_null = execute(
        agg(metric=Metric.COUNT, filters=[Filter(field="resolution_time_hrs", op=Op.NOT_NULL)]), store
    ).value
    assert is_null == 173
    assert not_null == 327
    assert is_null + not_null == store.row_count


def test_numeric_comparison_excludes_nulls(store):
    """NaN > 12 must be False, not an error and not a match."""
    r = execute(
        agg(metric=Metric.COUNT, filters=[Filter(field="resolution_time_hrs", op=Op.GT, value=0)]),
        store,
    )
    assert r.value == 327  # every non-null row, no unresolved tickets


def test_ne_on_nullable_column_excludes_nulls(store):
    r = execute(
        agg(metric=Metric.COUNT, filters=[Filter(field="customer_rating", op=Op.NE, value=5)]),
        store,
    )
    assert r.value == 327 - 95  # rated, but not 5


# -------------------------------------------------------------------- list


def test_list_returns_rows_and_caps_at_limit(store):
    r = execute(lst(filters=[Filter(field="priority", op=Op.EQ, value="Critical")], limit=5), store)
    assert r.matched_count == 55
    assert r.returned_count == 5
    assert len(r.rows) == 5
    assert "Showing the first 5" in r.answer


def test_list_hard_cap_applies_and_warns(store):
    r = execute(lst(limit=500), store)
    assert r.returned_count == 100
    assert any("capped at 100" in w for w in r.warnings)


def test_list_rows_are_json_safe(store):
    """Nulls must serialise as None, never NaN."""
    r = execute(lst(filters=[Filter(field="status", op=Op.EQ, value="Open")], limit=3), store)
    for row in r.rows:
        assert row["resolution_time_hrs"] is None
        assert row["customer_rating"] is None
        assert isinstance(row["created_at"], str)
        assert not any(isinstance(v, float) and math.isnan(v) for v in row.values())


def test_list_sorting_is_applied(store):
    r = execute(
        lst(filters=[Filter(field="status", op=Op.EQ, value="Resolved")],
            sort=Sort(field="resolution_time_hrs", direction="desc"), limit=3),
        store,
    )
    values = [row["resolution_time_hrs"] for row in r.rows]
    assert values == sorted(values, reverse=True)
    assert values[0] == 119.7


def test_empty_list_result_is_explained(store):
    r = execute(lst(filters=[Filter(field="issue_summary", op=Op.CONTAINS, value="zzzz")]), store)
    assert r.matched_count == 0
    assert r.rows == []
    assert "No tickets found" in r.answer
    assert "No tickets matched the filters." in r.warnings


# ---------------------------------------------------- safety & determinism


def test_executor_never_mutates_the_store(store):
    before = store.df.copy(deep=True)
    execute(lst(sort=Sort(field="resolution_time_hrs"), limit=10), store)
    execute(agg(metric=Metric.AVG, target="customer_rating", group_by="agent_id"), store)
    assert store.df.equals(before)
    assert len(store.df) == 500


def test_same_spec_produces_identical_results(store):
    spec = agg(metric=Metric.AVG, target="resolution_time_hrs", group_by="priority")
    first = execute(spec, store)
    second = execute(spec, store)
    assert first.model_dump() == second.model_dump()


def test_unsupported_intent_raises_with_reason(store):
    spec = QuerySpec(intent=Intent.UNSUPPORTED, reason="no sentiment column in this dataset")
    with pytest.raises(UnsupportedQueryError, match="sentiment"):
        execute(spec, store)


def test_anomaly_intent_is_routed_elsewhere(store):
    with pytest.raises(UnsupportedQueryError, match="anomaly detector"):
        execute(QuerySpec(intent=Intent.ANOMALY), store)


def test_result_is_fully_serialisable(store):
    r = execute(agg(metric=Metric.COUNT, group_by="priority"), store)
    dumped = r.model_dump(mode="json")
    assert dumped["groups"][0]["group"] == "Medium"
    assert dumped["intent"] == "aggregate"


# ------------------------------------------------- date-column filters (regression)
# Equality on a timestamp column compared a datetime to a date string and
# matched nothing. Silently returning zero rows is the worst failure mode here,
# so these lock in calendar-day semantics.


def test_eq_on_created_at_matches_the_calendar_day(store):
    r = execute(
        agg(metric=Metric.COUNT, filters=[Filter(field="created_at", op=Op.EQ, value="2024-03-05")]),
        store,
    )
    assert r.value == 7


def test_ne_on_created_at_excludes_only_that_day(store):
    r = execute(
        agg(metric=Metric.COUNT, filters=[Filter(field="created_at", op=Op.NE, value="2024-03-05")]),
        store,
    )
    assert r.value == 500 - 7


def test_in_on_created_at_spans_named_days(store):
    r = execute(
        agg(
            metric=Metric.COUNT,
            filters=[Filter(field="created_at", op=Op.IN, value=["2024-03-05", "2024-03-06"])],
        ),
        store,
    )
    assert r.value == 14


def test_unparseable_date_value_fails_loudly(store):
    spec = agg(metric=Metric.COUNT, filters=[Filter(field="created_at", op=Op.EQ, value="not-a-date")])
    with pytest.raises(UnsupportedQueryError, match="not a valid date"):
        execute(spec, store)


# --------------------------------------------- grouped sort direction (regression)
# Sorting a grouped result by the group key ignored `direction`, so "desc"
# silently returned ascending order.


def test_group_sort_by_key_honours_direction(store):
    asc = execute(
        agg(metric=Metric.COUNT, group_by="priority", sort=Sort(field="priority", direction="asc")),
        store,
    )
    desc = execute(
        agg(metric=Metric.COUNT, group_by="priority", sort=Sort(field="priority", direction="desc")),
        store,
    )
    assert [g.group for g in asc.groups] == ["Critical", "High", "Low", "Medium"]
    assert [g.group for g in desc.groups] == ["Medium", "Low", "High", "Critical"]


def test_group_sort_by_key_is_independent_of_value_order(store):
    """Sorting by key must not fall back to value ordering."""
    r = execute(
        agg(metric=Metric.COUNT, group_by="status", sort=Sort(field="status", direction="asc")),
        store,
    )
    assert [g.group for g in r.groups] == ["Escalated", "Open", "Resolved"]
    assert [g.value for g in r.groups] == [62, 111, 327]


# --------------------------------------------- grouped answer text (regression)
# The answer reported the truncated group count, implying only `limit` groups
# existed, and pluralised "status" as "statuss".


def test_group_count_reflects_all_groups_not_the_page(store):
    r = execute(
        agg(metric=Metric.COUNT, group_by="agent_id", sort=Sort(field="value"), limit=3),
        store,
    )
    assert r.group_count == 12
    assert len(r.groups) == 3
    assert "top 3 of 12 agents" in r.answer


def test_single_group_page_says_out_of_total(store):
    r = execute(
        agg(
            metric=Metric.AVG,
            target="customer_rating",
            group_by="agent_id",
            sort=Sort(field="value", direction="asc"),
            limit=1,
        ),
        store,
    )
    assert r.groups[0].group == "AGT-08"
    assert r.groups[0].value == 3.48
    assert "out of 12 agents" in r.answer
    assert "lowest average customer rating" in r.answer


def test_group_labels_are_pluralised_correctly(store):
    for group_by, plural in [
        ("status", "statuses"),
        ("category", "categories"),
        ("priority", "priorities"),
        ("agent_id", "agents"),
    ]:
        r = execute(agg(metric=Metric.COUNT, group_by=group_by), store)
        assert plural in r.answer, r.answer


def test_key_sorted_group_does_not_claim_highest(store):
    r = execute(
        agg(metric=Metric.COUNT, group_by="status", sort=Sort(field="status", direction="asc")),
        store,
    )
    assert "highest" not in r.answer and "lowest" not in r.answer
    assert "first by status" in r.answer


def test_grouped_n_used_counts_all_groups_not_the_page(store):
    """A limit trims the display, not the population the aggregate ran over."""
    full = execute(agg(metric=Metric.AVG, target="resolution_time_hrs",
                       group_by="agent_id"), store)
    limited = execute(agg(metric=Metric.AVG, target="resolution_time_hrs",
                          group_by="agent_id",
                          sort=Sort(field="value", direction="desc"), limit=3), store)
    assert full.n_used == 327
    assert limited.n_used == 327          # not the top-3 subtotal
    assert limited.returned_count == 3
    assert limited.group_count == 12
