"""Data layer tests, asserted against the real support_tickets.csv.

The expected values below were established by profiling the shipped dataset.
They are hard-coded deliberately: if the CSV is swapped, these should fail
loudly rather than adapt silently.
"""

import pandas as pd
import pytest

from app.data_store import SchemaError, load_data_store
from app.models import COLUMNS, NUMERIC_COLUMNS


def test_loads_all_rows(store):
    assert store.row_count == 500


def test_all_columns_present_and_typed(store):
    assert list(store.df.columns) == list(COLUMNS)
    assert pd.api.types.is_datetime64_any_dtype(store.df["created_at"])
    for col in NUMERIC_COLUMNS:
        assert pd.api.types.is_numeric_dtype(store.df[col])


def test_no_duplicate_ticket_ids(store):
    assert store.df["ticket_id"].duplicated().sum() == 0


def test_date_range_and_anchor(store):
    assert store.min_date == pd.Timestamp("2024-01-01 08:54")
    assert store.anchor == pd.Timestamp("2024-03-30 18:06")


def test_structural_nulls(store):
    """Nulls are structural: unresolved tickets carry no outcome."""
    assert store.null_counts["resolution_time_hrs"] == 173
    assert store.null_counts["customer_rating"] == 173
    assert store.null_counts["response_time_hrs"] == 0
    assert store.null_counts["created_at"] == 0


def test_nulls_align_exactly_with_status(store):
    df = store.df
    resolved = df["status"] == "Resolved"
    assert df.loc[resolved, "resolution_time_hrs"].notna().all()
    assert df.loc[resolved, "customer_rating"].notna().all()
    assert df.loc[~resolved, "resolution_time_hrs"].isna().all()
    assert df.loc[~resolved, "customer_rating"].isna().all()


def test_distinct_values(store):
    assert store.distinct_values["status"] == ["Escalated", "Open", "Resolved"]
    assert store.distinct_values["category"] == ["Billing", "General", "Technical"]
    assert sorted(store.distinct_values["priority"]) == [
        "Critical", "High", "Low", "Medium",
    ]
    assert len(store.distinct_values["agent_id"]) == 12


def test_integrity_contradiction_is_reported_not_dropped(store):
    """28 tickets are resolved before first response. Keep them, flag them."""
    contradictory = store.df["resolution_time_hrs"] < store.df["response_time_hrs"]
    assert int(contradictory.sum()) == 28
    assert store.row_count == 500  # not dropped
    assert any("resolved before first response" in w for w in store.load_warnings)


def test_metadata_is_serialisable(store):
    meta = store.metadata()
    assert meta["row_count"] == 500
    assert meta["date_anchor"] == "2024-03-30"
    assert meta["columns"]["resolution_time_hrs"] == "numeric"
    assert meta["columns"]["created_at"] == "datetime"
    assert meta["numeric_summary"]["resolution_time_hrs"]["max"] == 119.7


def test_missing_file_raises(tmp_path):
    with pytest.raises(SchemaError, match="not found"):
        load_data_store(tmp_path / "nope.csv")


def test_missing_column_raises(tmp_path, store):
    bad = tmp_path / "bad.csv"
    store.df.drop(columns=["priority"]).to_csv(bad, index=False)
    with pytest.raises(SchemaError, match="missing required column"):
        load_data_store(bad)


def test_unparseable_dates_are_dropped_with_warning(tmp_path, store):
    df = store.df.head(10).copy()
    df["created_at"] = df["created_at"].dt.strftime("%Y-%m-%d %H:%M")
    df.loc[0, "created_at"] = "not-a-date"
    path = tmp_path / "dates.csv"
    df.to_csv(path, index=False)

    loaded = load_data_store(path)
    assert loaded.row_count == 9
    assert any("unparseable" in w for w in loaded.load_warnings)


def test_unexpected_category_value_warns(tmp_path, store):
    df = store.df.head(10).copy()
    df["created_at"] = df["created_at"].dt.strftime("%Y-%m-%d %H:%M")
    df.loc[0, "priority"] = "Urgent"
    path = tmp_path / "priority.csv"
    df.to_csv(path, index=False)

    loaded = load_data_store(path)
    assert any("unexpected priority" in w for w in loaded.load_warnings)


def test_empty_dataset_raises(tmp_path, store):
    path = tmp_path / "empty.csv"
    store.df.head(0).to_csv(path, index=False)
    with pytest.raises(SchemaError, match="no usable rows"):
        load_data_store(path)
