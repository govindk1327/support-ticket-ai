"""Resolution of date presets into concrete half-open intervals.

Relative dates ("this month", "this week") resolve against the dataset's most
recent ticket, not wall-clock time. The dataset ends 2024-03-30; anchoring to
`datetime.now()` would make every relative query return zero rows. The anchor is
reported back to the caller so the interpretation is always visible.

The LLM emits only a preset name. Date arithmetic happens here, in Python,
because small models make quiet off-by-one errors on month and quarter
boundaries.
"""

from __future__ import annotations

import pandas as pd

from app.models import DatePreset, DateRange

# Intervals are half-open: start <= created_at < end. `None` means unbounded.
Interval = tuple[pd.Timestamp | None, pd.Timestamp | None]


def _month_start(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.normalize().replace(day=1)


def _add_months(ts: pd.Timestamp, n: int) -> pd.Timestamp:
    total = ts.month - 1 + n
    year = ts.year + total // 12
    month = total % 12 + 1
    return pd.Timestamp(year=year, month=month, day=1)


def _quarter_start(ts: pd.Timestamp) -> pd.Timestamp:
    first_month = 3 * ((ts.month - 1) // 3) + 1
    return pd.Timestamp(year=ts.year, month=first_month, day=1)


def resolve_preset(preset: DatePreset, anchor: pd.Timestamp) -> Interval:
    """Turn a preset into a half-open interval relative to `anchor`."""
    day = anchor.normalize()
    one_day = pd.Timedelta(days=1)

    if preset is DatePreset.ALL_TIME:
        return (None, None)
    if preset is DatePreset.TODAY:
        return (day, day + one_day)
    if preset is DatePreset.YESTERDAY:
        return (day - one_day, day)
    if preset is DatePreset.LAST_7_DAYS:
        return (day - pd.Timedelta(days=6), day + one_day)
    if preset is DatePreset.LAST_30_DAYS:
        return (day - pd.Timedelta(days=29), day + one_day)
    if preset is DatePreset.THIS_WEEK:
        start = day - pd.Timedelta(days=day.weekday())  # Monday
        return (start, start + pd.Timedelta(days=7))
    if preset is DatePreset.LAST_WEEK:
        this_week = day - pd.Timedelta(days=day.weekday())
        return (this_week - pd.Timedelta(days=7), this_week)
    if preset is DatePreset.THIS_MONTH:
        start = _month_start(day)
        return (start, _add_months(start, 1))
    if preset is DatePreset.LAST_MONTH:
        start = _add_months(_month_start(day), -1)
        return (start, _add_months(start, 1))
    if preset is DatePreset.THIS_QUARTER:
        start = _quarter_start(day)
        return (start, _add_months(start, 3))

    raise ValueError(f"unhandled date preset: {preset}")


def resolve(date_range: DateRange | None, anchor: pd.Timestamp) -> Interval:
    """Resolve a DateRange (preset or explicit ISO dates) to an interval."""
    if date_range is None:
        return (None, None)

    if date_range.preset is not None:
        return resolve_preset(date_range.preset, anchor)

    start = end = None
    if date_range.start:
        try:
            start = pd.Timestamp(date_range.start).normalize()
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid start date '{date_range.start}'") from exc
    if date_range.end:
        try:
            # `end` is inclusive of the whole day the user named.
            end = pd.Timestamp(date_range.end).normalize() + pd.Timedelta(days=1)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid end date '{date_range.end}'") from exc

    if start is not None and end is not None and start >= end:
        raise ValueError("start date must be before end date")

    return (start, end)


def describe(date_range: DateRange | None, interval: Interval) -> str:
    """Human-readable label for what the date filter actually did."""
    if date_range is None or interval == (None, None):
        return "all time"
    start, end = interval
    if date_range.preset is not None:
        label = date_range.preset.value.replace("_", " ")
    else:
        label = "custom range"
    if start is None:
        return f"{label} (up to {(end - pd.Timedelta(days=1)).date()})"
    if end is None:
        return f"{label} (from {start.date()})"
    return f"{label} ({start.date()} to {(end - pd.Timedelta(days=1)).date()})"
