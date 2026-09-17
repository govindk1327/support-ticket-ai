"""Schema constants and the validated intermediate representation (QuerySpec).

The LLM's only job is to emit a QuerySpec. It never emits code, SQL, or pandas
expressions. Everything the model produces is validated here before the executor
is allowed to touch the DataFrame.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --------------------------------------------------------------------------
# Dataset schema. Single source of truth for what the LLM is allowed to name.
# --------------------------------------------------------------------------

DATE_COLUMN = "created_at"

NUMERIC_COLUMNS = ("response_time_hrs", "resolution_time_hrs", "customer_rating")
STRING_COLUMNS = (
    "ticket_id",
    "category",
    "priority",
    "status",
    "agent_id",
    "issue_summary",
)
COLUMNS = STRING_COLUMNS + (DATE_COLUMN,) + NUMERIC_COLUMNS

# Columns that make sense as a group-by key. Grouping on ticket_id or a float
# column produces one group per row, which is never a useful answer.
GROUPABLE_COLUMNS = ("category", "priority", "status", "agent_id")

# Columns that may be null by design (unresolved tickets carry no outcome).
NULLABLE_COLUMNS = ("resolution_time_hrs", "customer_rating")

COLUMN_LABELS = {
    "ticket_id": "ticket ID",
    "created_at": "creation date",
    "category": "category",
    "priority": "priority",
    "status": "status",
    "response_time_hrs": "response time (hrs)",
    "resolution_time_hrs": "resolution time (hrs)",
    "agent_id": "agent",
    "customer_rating": "customer rating",
    "issue_summary": "issue summary",
}

# Plurals are explicit because naive "+ s" produces "statuss" / "categorys".
COLUMN_PLURALS = {
    "category": "categories",
    "priority": "priorities",
    "status": "statuses",
    "agent_id": "agents",
}

# Hard ceiling on rows returned by a `list` query, so a broad filter cannot
# dump the whole dataset into a response body.
MAX_LIST_LIMIT = 100
DEFAULT_LIST_LIMIT = 50


# --------------------------------------------------------------------------
# Enums. Anything outside these sets is rejected at parse time.
# --------------------------------------------------------------------------


class Intent(str, Enum):
    AGGREGATE = "aggregate"
    LIST = "list"
    ANOMALY = "anomaly"
    UNSUPPORTED = "unsupported"


class Metric(str, Enum):
    COUNT = "count"
    AVG = "avg"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"


class Op(str, Enum):
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    IS_NULL = "is_null"
    NOT_NULL = "not_null"


class DatePreset(str, Enum):
    TODAY = "today"
    YESTERDAY = "yesterday"
    THIS_WEEK = "this_week"
    LAST_WEEK = "last_week"
    LAST_7_DAYS = "last_7_days"
    THIS_MONTH = "this_month"
    LAST_MONTH = "last_month"
    LAST_30_DAYS = "last_30_days"
    THIS_QUARTER = "this_quarter"
    ALL_TIME = "all_time"


ORDERING_OPS = {Op.GT, Op.GTE, Op.LT, Op.LTE}
LIST_OPS = {Op.IN, Op.NOT_IN}
NULL_OPS = {Op.IS_NULL, Op.NOT_NULL}


# --------------------------------------------------------------------------
# Spec components
# --------------------------------------------------------------------------


class Filter(BaseModel):
    """A single predicate. Combined with others using AND."""

    model_config = ConfigDict(extra="forbid")

    field: str
    op: Op
    value: Any = None

    @field_validator("field")
    @classmethod
    def _known_column(cls, v: str) -> str:
        if v not in COLUMNS:
            raise ValueError(
                f"unknown column '{v}'; valid columns are {', '.join(COLUMNS)}"
            )
        return v

    @model_validator(mode="after")
    def _check_op_column_value(self) -> "Filter":
        col, op, val = self.field, self.op, self.value

        if op in NULL_OPS:
            if val is not None:
                raise ValueError(f"op '{op.value}' takes no value")
            return self

        if val is None:
            raise ValueError(f"op '{op.value}' requires a value")

        if op in LIST_OPS:
            if not isinstance(val, (list, tuple)) or len(val) == 0:
                raise ValueError(f"op '{op.value}' requires a non-empty list value")
            if col in NUMERIC_COLUMNS and not all(
                isinstance(x, (int, float)) and not isinstance(x, bool) for x in val
            ):
                raise ValueError(f"column '{col}' is numeric; list values must be numbers")
            if col == DATE_COLUMN and not all(isinstance(x, str) for x in val):
                raise ValueError(
                    f"column '{col}' is a date; list values must be 'YYYY-MM-DD' strings"
                )
            return self

        if isinstance(val, (list, tuple)):
            raise ValueError(f"op '{op.value}' takes a single value, not a list")

        if op in ORDERING_OPS and col not in NUMERIC_COLUMNS:
            raise ValueError(
                f"op '{op.value}' only applies to numeric columns "
                f"({', '.join(NUMERIC_COLUMNS)}); use a date_range for dates"
            )

        if op is Op.CONTAINS and col not in STRING_COLUMNS:
            raise ValueError(
                f"op 'contains' only applies to text columns "
                f"({', '.join(STRING_COLUMNS)})"
            )

        if col in NUMERIC_COLUMNS and not (
            isinstance(val, (int, float)) and not isinstance(val, bool)
        ):
            raise ValueError(f"column '{col}' is numeric; value must be a number")

        if col in STRING_COLUMNS and not isinstance(val, str):
            raise ValueError(f"column '{col}' is text; value must be a string")

        if col == DATE_COLUMN and not isinstance(val, str):
            raise ValueError(
                f"column '{col}' is a date; value must be a 'YYYY-MM-DD' string"
            )

        return self


class DateRange(BaseModel):
    """Either a named preset or an explicit inclusive ISO date range."""

    model_config = ConfigDict(extra="forbid")

    preset: DatePreset | None = None
    start: str | None = None
    end: str | None = None

    @model_validator(mode="after")
    def _one_form_only(self) -> "DateRange":
        has_explicit = self.start is not None or self.end is not None
        if self.preset is not None and has_explicit:
            raise ValueError("date_range takes either a preset or start/end, not both")
        if self.preset is None and not has_explicit:
            raise ValueError("date_range requires a preset or start/end")
        return self


class Sort(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    direction: Literal["asc", "desc"] = "desc"

    @field_validator("field")
    @classmethod
    def _sortable(cls, v: str) -> str:
        # "value" is the alias for the computed aggregate in a grouped result.
        if v != "value" and v not in COLUMNS:
            raise ValueError(f"cannot sort by '{v}'")
        return v


class QuerySpec(BaseModel):
    """The full validated interpretation of a user question."""

    model_config = ConfigDict(extra="forbid")

    intent: Intent
    metric: Metric | None = None
    target: str | None = None
    filters: list[Filter] = Field(default_factory=list)
    or_groups: list[list[Filter]] = Field(default_factory=list)
    group_by: str | None = None
    date_range: DateRange | None = None
    sort: Sort | None = None
    limit: int | None = Field(default=None, gt=0, le=1000)
    reason: str | None = None

    @field_validator("target")
    @classmethod
    def _target_known(cls, v: str | None) -> str | None:
        if v is not None and v not in COLUMNS:
            raise ValueError(f"unknown target column '{v}'")
        return v

    @field_validator("group_by")
    @classmethod
    def _groupable(cls, v: str | None) -> str | None:
        if v is not None and v not in GROUPABLE_COLUMNS:
            raise ValueError(
                f"cannot group by '{v}'; groupable columns are "
                f"{', '.join(GROUPABLE_COLUMNS)}"
            )
        return v

    @model_validator(mode="after")
    def _coherent(self) -> "QuerySpec":
        if self.intent is Intent.UNSUPPORTED:
            if not self.reason:
                raise ValueError("unsupported queries must explain why in 'reason'")
            return self

        if self.intent is Intent.AGGREGATE:
            if self.metric is None:
                raise ValueError("aggregate queries require a metric")
            if self.metric is Metric.COUNT:
                self.target = None  # count ignores target; drop it rather than fail
            else:
                if self.target is None:
                    raise ValueError(f"metric '{self.metric.value}' requires a target column")
                if self.target not in NUMERIC_COLUMNS:
                    raise ValueError(
                        f"metric '{self.metric.value}' needs a numeric target; "
                        f"'{self.target}' is not numeric"
                    )
        else:
            # list / anomaly carry no metric
            if self.metric is not None:
                raise ValueError(f"intent '{self.intent.value}' does not take a metric")

        if self.sort and self.sort.field == "value" and self.group_by is None:
            raise ValueError("sort by 'value' is only valid with group_by")

        # A grouped result only has two things to sort on: the computed value or
        # the group key. Sorting it by any other column is meaningless, so it is
        # rejected here rather than silently ignored by the executor.
        if self.group_by is not None and self.sort is not None:
            if self.sort.field not in ("value", self.group_by):
                raise ValueError(
                    f"a grouped result can only be sorted by 'value' or by the "
                    f"group key '{self.group_by}', not '{self.sort.field}'"
                )

        if self.group_by is not None and self.intent is not Intent.AGGREGATE:
            raise ValueError("group_by is only valid for aggregate queries")

        for group in self.or_groups:
            if len(group) < 2:
                raise ValueError("each or_group needs at least two filters")

        return self


# --------------------------------------------------------------------------
# Result models
# --------------------------------------------------------------------------


class GroupResult(BaseModel):
    group: str
    value: float | int | None
    n_used: int
    n_total: int


class AppliedDateRange(BaseModel):
    label: str
    start: str | None = None
    end: str | None = None


class QueryResult(BaseModel):
    """Everything the executor computed. All numbers come from pandas."""

    intent: Intent
    metric: Metric | None = None
    target: str | None = None
    value: float | int | None = None
    groups: list[GroupResult] | None = None
    # Total groups before `limit` truncated the list, so the answer text can say
    # "top 3 of 12" rather than implying only 3 groups existed.
    group_count: int = 0
    rows: list[dict] | None = None
    columns: list[str] | None = None
    matched_count: int = 0
    returned_count: int = 0
    n_used: int = 0
    n_total: int = 0
    date_range: AppliedDateRange | None = None
    answer: str = ""
    warnings: list[str] = Field(default_factory=list)


class UnsupportedQueryError(Exception):
    """Raised when a spec cannot be executed against this dataset."""


class AnomalyFilterError(UnsupportedQueryError):
    """A requested filter cannot be safely applied to anomaly detection.

    Raised instead of silently ignoring the filter. Filtering on the very
    column a rule measures makes that rule circular: narrowing to
    `resolution_time_hrs > 50` and then asking which resolution times are
    outliers computes the threshold from an already-truncated population.
    """

    def __init__(self, field: str, reason: str, allowed: list[str] | None = None):
        self.field = field
        self.reason = reason
        self.allowed = list(allowed or [])
        super().__init__(
            f"Cannot filter anomaly detection on '{field}'. {reason} "
            f"Filter instead on: {', '.join(self.allowed)}."
        )

    def to_dict(self) -> dict:
        return {
            "error": "unsupported_anomaly_filter",
            "field": self.field,
            "reason": self.reason,
            "allowed_filter_fields": self.allowed,
            "message": str(self),
        }


class ValueNormalizationError(UnsupportedQueryError):
    """A categorical filter value does not resolve to a real dataset value.

    Subclasses UnsupportedQueryError so existing handlers keep working, but
    carries structured fields so the Part 3 repair retry can feed the model the
    column, the rejected value, and the list of values it may actually use.
    """

    def __init__(
        self,
        field: str,
        value: object,
        valid_values: list[str],
        reason: str = "unknown",
        candidates: list[str] | None = None,
    ):
        self.field = field
        self.value = value
        self.valid_values = list(valid_values)
        self.reason = reason  # "unknown" | "ambiguous"
        self.candidates = list(candidates or [])

        options = ", ".join(self.valid_values)
        if reason == "ambiguous":
            matched = ", ".join(self.candidates)
            message = (
                f"'{value}' is ambiguous for column '{field}'; it matches "
                f"{matched}. Valid values are: {options}"
            )
        else:
            message = (
                f"'{value}' is not a valid value for column '{field}'. "
                f"Valid values are: {options}"
            )
        super().__init__(message)

    def to_dict(self) -> dict:
        """Machine-readable form for API responses and repair prompts."""
        return {
            "error": "invalid_filter_value",
            "field": self.field,
            "value": self.value,
            "reason": self.reason,
            "valid_values": self.valid_values,
            "candidates": self.candidates,
            "message": str(self),
        }


# --------------------------------------------------------------------------
# Anomaly detection
# --------------------------------------------------------------------------


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Rule(str, Enum):
    """Stable rule identifiers.

    The API filter, the UI and the tests all reference these exact strings, so
    they are an enum rather than loose text.
    """

    RESOLUTION_TIME_OUTLIER = "resolution_time_outlier"
    UNRESOLVED_HIGH_PRIORITY = "unresolved_high_priority"
    RESPONSE_SLA_BREACH = "response_sla_breach"
    DATA_INTEGRITY_CONTRADICTION = "data_integrity_contradiction"


class AnomalyFlag(BaseModel):
    """One flagged ticket.

    Always carries the observed value and the threshold it crossed, so the
    explanation is checkable rather than asserted. This is why the detector is
    rules-based: an unsupervised score cannot produce this record.
    """

    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    rule: Rule
    severity: Severity
    metric: str = Field(..., description="Column the observed value came from.")
    observed_value: float
    threshold: float
    explanation: str

    # Context so the UI table is readable without a second lookup.
    priority: str | None = None
    status: str | None = None
    category: str | None = None
    agent_id: str | None = None
    created_at: str | None = None


class AnomalyReport(BaseModel):
    flags: list[AnomalyFlag] = Field(default_factory=list)
    total_flagged: int = 0
    returned_count: int = 0
    tickets_flagged: int = 0
    # Full-dataset counts, unaffected by filters or limit.
    summary: dict[str, int] = Field(default_factory=dict)
    severity_summary: dict[str, int] = Field(default_factory=dict)
    # Counts for the filtered set, before `limit` truncates it.
    matched_summary: dict[str, int] = Field(default_factory=dict)
    matched_severity_summary: dict[str, int] = Field(default_factory=dict)
    thresholds: dict = Field(default_factory=dict)
    date_anchor: str | None = None
    # Human description of the population the rules ran over, when filtered.
    scope: str | None = None
    notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
