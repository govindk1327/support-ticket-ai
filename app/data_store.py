"""CSV ingestion and the in-memory dataset.

Loaded once at startup. 500 rows fits comfortably in memory, so pandas is the
right tool; nothing here survives a restart and that is an accepted trade for
this assessment (see README limitations).

The store is treated as read-only after load. The executor never mutates it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from app.models import (
    COLUMNS,
    DATE_COLUMN,
    GROUPABLE_COLUMNS,
    NUMERIC_COLUMNS,
    STRING_COLUMNS,
    Filter,
    Op,
    QuerySpec,
    ValueNormalizationError,
)

DEFAULT_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "support_tickets.csv"

# Values we expect in the categorical columns. Anything else is surfaced as a
# load warning rather than silently accepted.
EXPECTED_VALUES = {
    "category": {"Billing", "Technical", "General"},
    "priority": {"Low", "Medium", "High", "Critical"},
    "status": {"Open", "Resolved", "Escalated"},
}


class SchemaError(Exception):
    """The CSV does not match the expected schema and cannot be used."""


class DataStore:
    """Holds the ticket DataFrame plus the metadata derived from it."""

    def __init__(self, df: pd.DataFrame, warnings: list[str], source: str):
        self.df = df
        self.load_warnings = warnings
        self.source = source

        self.anchor: pd.Timestamp = df[DATE_COLUMN].max()
        self.min_date: pd.Timestamp = df[DATE_COLUMN].min()
        self.row_count: int = len(df)

        self.distinct_values: dict[str, list[str]] = {
            col: sorted(df[col].dropna().unique().tolist()) for col in GROUPABLE_COLUMNS
        }
        self.null_counts: dict[str, int] = {
            col: int(df[col].isna().sum()) for col in COLUMNS
        }

    # -- metadata -----------------------------------------------------------

    def numeric_summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for col in NUMERIC_COLUMNS:
            s = self.df[col].dropna()
            if s.empty:
                continue
            out[col] = {
                "count": int(s.count()),
                "min": float(s.min()),
                "max": float(s.max()),
                "mean": round(float(s.mean()), 2),
                "median": float(s.median()),
            }
        return out

    def metadata(self) -> dict:
        """Everything a caller (or a prompt) needs to know about the dataset."""
        return {
            "source": self.source,
            "row_count": self.row_count,
            "date_range": {
                "min": self.min_date.isoformat(),
                "max": self.anchor.isoformat(),
            },
            "date_anchor": self.anchor.normalize().date().isoformat(),
            "columns": {
                col: ("datetime" if col == DATE_COLUMN
                      else "numeric" if col in NUMERIC_COLUMNS
                      else "string")
                for col in COLUMNS
            },
            "distinct_values": self.distinct_values,
            "null_counts": self.null_counts,
            "numeric_summary": self.numeric_summary(),
            "load_warnings": self.load_warnings,
        }


def _coerce(df: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    df = df.copy()

    parsed = pd.to_datetime(df[DATE_COLUMN], format="%Y-%m-%d %H:%M", errors="coerce")
    unparsed = int(parsed.isna().sum())
    if unparsed:
        warnings.append(f"{unparsed} row(s) had an unparseable {DATE_COLUMN} and were dropped")
    df[DATE_COLUMN] = parsed

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in STRING_COLUMNS:
        df[col] = df[col].astype("string").str.strip()

    df = df[df[DATE_COLUMN].notna()].reset_index(drop=True)
    return df


def _validate(df: pd.DataFrame, warnings: list[str]) -> None:
    """Non-fatal integrity checks. Findings become warnings, not exceptions."""
    dupes = int(df["ticket_id"].duplicated().sum())
    if dupes:
        warnings.append(f"{dupes} duplicate ticket_id value(s)")

    for col, expected in EXPECTED_VALUES.items():
        unexpected = set(df[col].dropna().unique()) - expected
        if unexpected:
            warnings.append(f"unexpected {col} value(s): {', '.join(sorted(unexpected))}")

    # Resolved tickets should carry an outcome; unresolved ones should not.
    resolved = df["status"] == "Resolved"
    missing_outcome = int((resolved & df["resolution_time_hrs"].isna()).sum())
    if missing_outcome:
        warnings.append(f"{missing_outcome} resolved ticket(s) missing resolution_time_hrs")
    stray_outcome = int((~resolved & df["resolution_time_hrs"].notna()).sum())
    if stray_outcome:
        warnings.append(f"{stray_outcome} unresolved ticket(s) carry a resolution_time_hrs")

    # Resolved before first response is impossible; flagged here and again as an
    # anomaly rule in the detector.
    contradictory = int((df["resolution_time_hrs"] < df["response_time_hrs"]).sum())
    if contradictory:
        warnings.append(
            f"{contradictory} ticket(s) have resolution_time_hrs < response_time_hrs "
            "(resolved before first response)"
        )

    ratings = df["customer_rating"].dropna()
    if not ratings.empty and (ratings.lt(1).any() or ratings.gt(5).any()):
        warnings.append("customer_rating values outside the 1-5 range")


def load_data_store(csv_path: str | Path | None = None) -> DataStore:
    """Load and validate the ticket CSV. Raises SchemaError if unusable."""
    path = Path(csv_path or os.getenv("CSV_PATH") or DEFAULT_CSV_PATH)

    if not path.exists():
        raise SchemaError(f"dataset not found at {path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - surfaced to the caller as-is
        raise SchemaError(f"could not read {path}: {exc}") from exc

    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required column(s): {', '.join(missing)}")

    warnings: list[str] = []
    extra = [c for c in df.columns if c not in COLUMNS]
    if extra:
        warnings.append(f"ignoring unexpected column(s): {', '.join(extra)}")

    df = _coerce(df[list(COLUMNS)], warnings)
    if df.empty:
        raise SchemaError("dataset contains no usable rows after parsing")

    _validate(df, warnings)
    return DataStore(df, warnings, str(path))


# --------------------------------------------------------------------------
# Filter value normalisation
#
# The model writes "critical" or "tech"; the dataset holds "Critical" and
# "Technical". Resolving these against the real values means a near-miss gets
# answered correctly, and a genuine miss ("Urgent") is rejected with the valid
# options instead of silently returning zero tickets -- which reads like a real
# answer and is the worst failure mode this system has.
#
# Matching is exact, then case-insensitive, then unique prefix. Deliberately no
# edit-distance matching: "Hugh" -> "High" would be a guess, and a wrong guess
# here produces a confident wrong number.
# --------------------------------------------------------------------------

# Only these columns have a closed set of values worth normalising against.
# `contains` is excluded everywhere: substring search is meant to be partial.
NORMALISED_OPS = {Op.EQ, Op.NE, Op.IN, Op.NOT_IN}


def _resolve_value(field: str, value: str, valid: list[str]) -> str:
    """Resolve one raw value to its canonical dataset form."""
    if not isinstance(value, str):
        raise ValueNormalizationError(field, value, valid, reason="unknown")

    if value in valid:  # 1. exact
        return value

    lowered = value.lower()

    exact_ci = [v for v in valid if v.lower() == lowered]
    if len(exact_ci) == 1:  # 2. case-insensitive
        return exact_ci[0]

    prefixed = [v for v in valid if v.lower().startswith(lowered)]
    if len(prefixed) == 1:  # 3. unique prefix
        return prefixed[0]
    if len(prefixed) > 1:
        raise ValueNormalizationError(
            field, value, valid, reason="ambiguous", candidates=sorted(prefixed)
        )

    raise ValueNormalizationError(field, value, valid, reason="unknown")  # 4. reject


def _normalise_filter(f: Filter, store: "DataStore") -> Filter:
    valid = store.distinct_values.get(f.field)
    if valid is None or f.op not in NORMALISED_OPS:
        return f

    if isinstance(f.value, (list, tuple)):
        resolved = [_resolve_value(f.field, v, valid) for v in f.value]
    else:
        resolved = _resolve_value(f.field, f.value, valid)

    if resolved == f.value:
        return f
    return f.model_copy(update={"value": resolved})


def normalize_spec(spec: QuerySpec, store: "DataStore") -> QuerySpec:
    """Return a copy of `spec` with categorical filter values canonicalised.

    Raises ValueNormalizationError if any value cannot be resolved. The spec is
    never mutated in place, so execution stays deterministic and repeatable.
    """
    filters = [_normalise_filter(f, store) for f in spec.filters]
    or_groups = [[_normalise_filter(f, store) for f in g] for g in spec.or_groups]

    if filters == list(spec.filters) and or_groups == [list(g) for g in spec.or_groups]:
        return spec
    return spec.model_copy(update={"filters": filters, "or_groups": or_groups})
