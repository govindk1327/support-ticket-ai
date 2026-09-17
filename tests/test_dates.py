"""Date preset resolution.

Anchor for these tests is 2024-03-30 18:06 (a Saturday), the dataset's most
recent ticket. Intervals are half-open: start <= created_at < end.
"""

import pandas as pd
import pytest

from app.models import DatePreset, DateRange
from app.query import dates


def ts(s):
    return pd.Timestamp(s)


@pytest.mark.parametrize(
    "preset,expected_start,expected_end",
    [
        (DatePreset.TODAY, "2024-03-30", "2024-03-31"),
        (DatePreset.YESTERDAY, "2024-03-29", "2024-03-30"),
        (DatePreset.LAST_7_DAYS, "2024-03-24", "2024-03-31"),
        (DatePreset.LAST_30_DAYS, "2024-03-01", "2024-03-31"),
        # Saturday -> week starts Monday 2024-03-25
        (DatePreset.THIS_WEEK, "2024-03-25", "2024-04-01"),
        (DatePreset.LAST_WEEK, "2024-03-18", "2024-03-25"),
        (DatePreset.THIS_MONTH, "2024-03-01", "2024-04-01"),
        (DatePreset.LAST_MONTH, "2024-02-01", "2024-03-01"),
        (DatePreset.THIS_QUARTER, "2024-01-01", "2024-04-01"),
    ],
)
def test_presets(preset, expected_start, expected_end, fixed_anchor):
    start, end = dates.resolve_preset(preset, fixed_anchor)
    assert start == ts(expected_start)
    assert end == ts(expected_end)


def test_all_time_is_unbounded(fixed_anchor):
    assert dates.resolve_preset(DatePreset.ALL_TIME, fixed_anchor) == (None, None)


def test_no_date_range_is_unbounded(fixed_anchor):
    assert dates.resolve(None, fixed_anchor) == (None, None)


def test_year_boundary_for_last_month():
    start, end = dates.resolve_preset(DatePreset.LAST_MONTH, ts("2024-01-15"))
    assert start == ts("2023-12-01")
    assert end == ts("2024-01-01")


def test_leap_year_february():
    start, end = dates.resolve_preset(DatePreset.THIS_MONTH, ts("2024-02-29 10:00"))
    assert (start, end) == (ts("2024-02-01"), ts("2024-03-01"))


def test_quarter_boundaries():
    for day, q_start, q_end in [
        ("2024-01-01", "2024-01-01", "2024-04-01"),
        ("2024-04-01", "2024-04-01", "2024-07-01"),
        ("2024-12-31", "2024-10-01", "2025-01-01"),
    ]:
        assert dates.resolve_preset(DatePreset.THIS_QUARTER, ts(day)) == (
            ts(q_start), ts(q_end),
        )


def test_explicit_range_end_is_inclusive_of_the_day(fixed_anchor):
    start, end = dates.resolve(
        DateRange(start="2024-01-05", end="2024-01-10"), fixed_anchor
    )
    assert start == ts("2024-01-05")
    assert end == ts("2024-01-11")  # exclusive bound covers all of the 10th


def test_open_ended_explicit_ranges(fixed_anchor):
    assert dates.resolve(DateRange(start="2024-02-01"), fixed_anchor)[1] is None
    assert dates.resolve(DateRange(end="2024-02-01"), fixed_anchor)[0] is None


def test_inverted_range_rejected(fixed_anchor):
    with pytest.raises(ValueError, match="before end date"):
        dates.resolve(DateRange(start="2024-03-01", end="2024-02-01"), fixed_anchor)


def test_invalid_date_string_rejected(fixed_anchor):
    with pytest.raises(ValueError, match="invalid start date"):
        dates.resolve(DateRange(start="last tuesday"), fixed_anchor)


def test_date_range_requires_one_form():
    with pytest.raises(ValueError):
        DateRange()
    with pytest.raises(ValueError, match="not both"):
        DateRange(preset=DatePreset.THIS_MONTH, start="2024-01-01")


def test_describe_mentions_resolved_window(fixed_anchor):
    dr = DateRange(preset=DatePreset.THIS_MONTH)
    label = dates.describe(dr, dates.resolve(dr, fixed_anchor))
    assert "this month" in label
    assert "2024-03-01" in label and "2024-03-31" in label


def test_anchor_is_dataset_not_wallclock(store):
    """The key decision: relative dates anchor to the data, not to today."""
    start, end = dates.resolve_preset(DatePreset.THIS_MONTH, store.anchor)
    in_window = store.df[
        (store.df["created_at"] >= start) & (store.df["created_at"] < end)
    ]
    assert len(in_window) > 0, "this_month must return rows on a historical dataset"
