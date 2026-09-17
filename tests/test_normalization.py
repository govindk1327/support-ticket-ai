"""Categorical filter value normalisation.

The model writes "critical" or "tech"; the dataset holds "Critical" and
"Technical". Resolving near-misses makes those queries work, and rejecting
genuine misses stops the system answering "0 tickets" to a question about a
priority that does not exist -- a wrong answer that looks exactly like a real
one.

Matching is exact, then case-insensitive, then unique prefix. No edit-distance
matching, by design: a guess here becomes a confident wrong number.
"""

import pytest

from app.data_store import normalize_spec
from app.models import (
    Filter,
    Intent,
    Metric,
    Op,
    QuerySpec,
    Sort,
    ValueNormalizationError,
)
from app.query.executor import execute


def count(**kw):
    return QuerySpec(intent=Intent.AGGREGATE, metric=Metric.COUNT, **kw)


def one(field, value, op=Op.EQ):
    return count(filters=[Filter(field=field, op=op, value=value)])


def value_of(spec, store, index=0):
    return normalize_spec(spec, store).filters[index].value


# ------------------------------------------------------------- 1. exact match


def test_exact_value_is_accepted_unchanged(store):
    spec = one("priority", "Critical")
    assert value_of(spec, store) == "Critical"
    assert execute(spec, store).value == 55


def test_exact_match_returns_the_same_spec_object(store):
    """Nothing to change means no copy, so execution stays cheap."""
    spec = one("status", "Open")
    assert normalize_spec(spec, store) is spec


# -------------------------------------------------------- 2. case-insensitive


@pytest.mark.parametrize(
    "field,written,canonical,expected",
    [
        ("priority", "critical", "Critical", 55),
        ("priority", "CRITICAL", "Critical", 55),
        ("status", "open", "Open", 111),
        ("status", "OPEN", "Open", 111),
        ("category", "TECHNICAL", "Technical", 152),
        ("agent_id", "agt-04", "AGT-04", 37),
    ],
)
def test_case_insensitive_values_are_canonicalised(
    store, field, written, canonical, expected
):
    spec = one(field, written)
    assert value_of(spec, store) == canonical
    assert execute(spec, store).value == expected


# ------------------------------------------------------------ 3. unique prefix


@pytest.mark.parametrize(
    "field,written,canonical,expected",
    [
        ("category", "tech", "Technical", 152),
        ("category", "bill", "Billing", 159),
        ("priority", "crit", "Critical", 55),
        ("priority", "c", "Critical", 55),
        ("status", "res", "Resolved", 327),
        ("status", "esc", "Escalated", 62),
    ],
)
def test_unique_prefix_resolves_to_canonical_value(
    store, field, written, canonical, expected
):
    spec = one(field, written)
    assert value_of(spec, store) == canonical
    assert execute(spec, store).value == expected


def test_prefix_match_is_case_insensitive(store):
    assert value_of(one("category", "TECH"), store) == "Technical"


# ---------------------------------------------------------- 4. unknown values


@pytest.mark.parametrize(
    "field,written",
    [
        ("priority", "Urgent"),
        ("priority", "P1"),
        ("category", "Sales"),
        ("status", "Closed"),
        ("agent_id", "AGT-99"),
    ],
)
def test_unknown_value_is_rejected(store, field, written):
    with pytest.raises(ValueNormalizationError) as exc:
        execute(one(field, written), store)
    assert exc.value.reason == "unknown"
    assert exc.value.field == field


def test_rejection_names_the_valid_values(store):
    with pytest.raises(ValueNormalizationError) as exc:
        execute(one("priority", "Urgent"), store)
    message = str(exc.value)
    for valid in ("Critical", "High", "Low", "Medium"):
        assert valid in message


def test_unknown_value_does_not_silently_return_zero(store):
    """The whole point: 'Urgent' must not answer '0 tickets'."""
    with pytest.raises(ValueNormalizationError):
        execute(one("priority", "Urgent"), store)


def test_no_fuzzy_matching_on_typos(store):
    """Edit-distance matching is deliberately absent; a typo is rejected."""
    for typo in ("Hugh", "Criticl", "Tehnical"):
        with pytest.raises(ValueNormalizationError):
            execute(one("priority" if typo != "Tehnical" else "category", typo), store)


# -------------------------------------------------------- 5. ambiguous prefix


def test_ambiguous_prefix_is_rejected_not_guessed(store):
    with pytest.raises(ValueNormalizationError) as exc:
        execute(one("agent_id", "AGT-1"), store)
    assert exc.value.reason == "ambiguous"
    assert exc.value.candidates == ["AGT-10", "AGT-11", "AGT-12"]


def test_ambiguous_error_lists_the_matches(store):
    with pytest.raises(ValueNormalizationError) as exc:
        execute(one("agent_id", "AGT-0"), store)
    assert exc.value.reason == "ambiguous"
    assert len(exc.value.candidates) == 9


def test_longer_prefix_disambiguates(store):
    """AGT-1 is ambiguous; AGT-11 is not."""
    assert value_of(one("agent_id", "agt-11"), store) == "AGT-11"


# ------------------------------------------- repair-loop error information


def test_error_payload_is_machine_readable(store):
    with pytest.raises(ValueNormalizationError) as exc:
        execute(one("category", "Sales"), store)
    payload = exc.value.to_dict()
    assert payload["error"] == "invalid_filter_value"
    assert payload["field"] == "category"
    assert payload["value"] == "Sales"
    assert payload["reason"] == "unknown"
    assert payload["valid_values"] == ["Billing", "General", "Technical"]
    assert "Billing" in payload["message"]


def test_error_is_catchable_as_unsupported_query(store):
    """Subclassing keeps existing handlers working."""
    from app.models import UnsupportedQueryError

    with pytest.raises(UnsupportedQueryError):
        execute(one("priority", "Urgent"), store)


# ------------------------------------------------------- scope of normalisation


def test_all_normalised_operators_are_covered(store):
    assert value_of(one("priority", "crit", Op.NE), store) == "Critical"
    assert value_of(one("priority", ["crit", "hi"], Op.IN), store) == ["Critical", "High"]
    assert value_of(one("category", ["tech"], Op.NOT_IN), store) == ["Technical"]


def test_bad_value_inside_a_list_is_rejected(store):
    with pytest.raises(ValueNormalizationError) as exc:
        execute(one("priority", ["Critical", "Urgent"], Op.IN), store)
    assert exc.value.value == "Urgent"


def test_contains_is_not_normalised(store):
    """Substring search is meant to be partial; leave it alone."""
    spec = one("issue_summary", "login", Op.CONTAINS)
    assert value_of(spec, store) == "login"
    assert execute(spec, store).value > 0


def test_free_text_and_numeric_columns_are_untouched(store):
    assert value_of(one("ticket_id", "TKT-001"), store) == "TKT-001"
    assert value_of(one("resolution_time_hrs", 12, Op.GT), store) == 12


def test_or_group_values_are_normalised(store):
    """Sample query 3 still works when the model writes lowercase."""
    spec = QuerySpec(
        intent=Intent.LIST,
        filters=[Filter(field="priority", op=Op.EQ, value="crit")],
        or_groups=[
            [
                Filter(field="status", op=Op.NE, value="resolved"),
                Filter(field="resolution_time_hrs", op=Op.GT, value=12),
            ]
        ],
    )
    normalised = normalize_spec(spec, store)
    assert normalised.filters[0].value == "Critical"
    assert normalised.or_groups[0][0].value == "Resolved"
    assert execute(spec, store).matched_count == 34


def test_normalisation_does_not_mutate_the_input_spec(store):
    spec = one("priority", "crit")
    normalize_spec(spec, store)
    assert spec.filters[0].value == "crit"


# ------------------------------------------------ existing queries unchanged


def test_canonical_sample_queries_are_unaffected(store):
    """Every already-valid query must return exactly what it did before."""
    assert execute(one("status", "Open"), store).value == 111
    assert execute(
        count(
            filters=[
                Filter(field="category", op=Op.EQ, value="Billing"),
                Filter(field="priority", op=Op.EQ, value="High"),
            ]
        ),
        store,
    ).value == 45

    avg = QuerySpec(
        intent=Intent.AGGREGATE,
        metric=Metric.AVG,
        target="customer_rating",
        filters=[Filter(field="category", op=Op.EQ, value="Technical")],
    )
    r = execute(avg, store)
    assert r.value == 3.74 and r.n_used == 104 and r.n_total == 152

    grouped = execute(
        count(group_by="agent_id", sort=Sort(field="value", direction="desc"), limit=3),
        store,
    )
    assert grouped.group_count == 12
    assert grouped.groups[0].group == "AGT-09"


def test_written_and_canonical_forms_agree(store):
    """A normalised query and its canonical twin produce identical results."""
    written = execute(one("category", "tech"), store)
    canonical = execute(one("category", "Technical"), store)
    assert written.value == canonical.value
    assert written.answer == canonical.answer
