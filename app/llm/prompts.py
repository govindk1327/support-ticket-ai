"""Prompt construction for question -> QuerySpec translation.

The system prompt is built from the loaded DataStore rather than hardcoded, so
the schema, the real categorical values and the date anchor come from the same
place the executor reads them. If the CSV changes, the prompt changes with it.

Two things the model is explicitly NOT asked to do:
  - compute dates (it emits a preset name; app/query/dates.py does the arithmetic)
  - produce code, SQL or pandas expressions (it emits a constrained spec)
"""

from __future__ import annotations

import json

from app.models import (
    COLUMNS,
    DATE_COLUMN,
    GROUPABLE_COLUMNS,
    NULLABLE_COLUMNS,
    NUMERIC_COLUMNS,
    DatePreset,
    Metric,
    Op,
)

SPEC_SHAPE = """{
  "intent":     "aggregate" | "list" | "anomaly" | "unsupported",
  "metric":     "count" | "avg" | "sum" | "min" | "max" | "median",
  "target":     numeric column name (required for every metric except count),
  "filters":    [ {"field": col, "op": operator, "value": v} ],
  "or_groups":  [ [ {...}, {...} ] ],
  "group_by":   categorical column name or null,
  "date_range": {"preset": name} or {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"},
  "sort":       {"field": "value" | group column, "direction": "asc" | "desc"},
  "limit":      integer,
  "reason":     string (required when intent is "unsupported")
}"""

RULES = """RULES
1. Output one JSON object and nothing else. No prose, no markdown fences.
2. Use only the column names listed above. Never invent a column.
3. filters are combined with AND. Use or_groups when a condition is genuinely
   an OR; each group needs at least two filters and is ANDed with the rest.
4. Never compute dates yourself. Emit a date_range preset and the system
   resolves it. Relative phrases like "this month" map to presets.
5. Use date_range for time windows, never a comparison operator on created_at.
   eq/ne/in on created_at mean "on that calendar day".
6. count ignores target. avg/sum/min/max/median need a numeric target.
7. group_by is only valid with intent "aggregate", and only on the categorical
   columns listed. To find the top/bottom group, set group_by plus
   sort {"field": "value"} and limit 1.
8. "not resolved within N hours" means status is not Resolved OR
   resolution_time_hrs > N. Use an or_group; an AND-only reading silently drops
   every still-open ticket.
9. If the question needs data this dataset does not contain, or an operation
   this schema cannot express, return intent "unsupported" with a reason that
   says what is missing. Do not force a wrong query.
10. Filter values must be real dataset values from the lists above."""


def _column_lines(store) -> str:
    kinds = {
        DATE_COLUMN: "datetime (YYYY-MM-DD HH:MM)",
        **{c: "numeric" for c in NUMERIC_COLUMNS},
    }
    lines = []
    for col in COLUMNS:
        kind = kinds.get(col, "text")
        notes = []
        if col in store.distinct_values:
            notes.append("values: " + ", ".join(store.distinct_values[col]))
        if col in NULLABLE_COLUMNS:
            notes.append(f"null for unresolved tickets ({store.null_counts[col]} rows)")
        suffix = f" -- {'; '.join(notes)}" if notes else ""
        lines.append(f"  {col} ({kind}){suffix}")
    return "\n".join(lines)


# Few-shot examples. Deliberately not verbatim copies of the assessment's sample
# queries -- they cover the same shapes in different wording, so the prompt
# demonstrates the pattern without teaching the answers to the test.
EXAMPLES: list[tuple[str, dict]] = [
    (
        "How many escalated tickets are there?",
        {"intent": "aggregate", "metric": "count",
         "filters": [{"field": "status", "op": "eq", "value": "Escalated"}]},
    ),
    (
        "What's the median resolution time for billing issues last month?",
        {"intent": "aggregate", "metric": "median", "target": "resolution_time_hrs",
         "filters": [{"field": "category", "op": "eq", "value": "Billing"}],
         "date_range": {"preset": "last_month"}},
    ),
    (
        "Which agent has the lowest average customer rating?",
        {"intent": "aggregate", "metric": "avg", "target": "customer_rating",
         "group_by": "agent_id",
         "sort": {"field": "value", "direction": "asc"}, "limit": 1},
    ),
    (
        "List high priority tickets that weren't closed within 8 hours",
        {"intent": "list",
         "filters": [{"field": "priority", "op": "eq", "value": "High"}],
         "or_groups": [[
             {"field": "status", "op": "ne", "value": "Resolved"},
             {"field": "resolution_time_hrs", "op": "gt", "value": 8},
         ]]},
    ),
    (
        "Break down ticket counts by category this quarter",
        {"intent": "aggregate", "metric": "count", "group_by": "category",
         "date_range": {"preset": "this_quarter"},
         "sort": {"field": "value", "direction": "desc"}},
    ),
    (
        "Any unusual resolution times recently?",
        {"intent": "anomaly", "date_range": {"preset": "last_7_days"}},
    ),
    (
        "Which tickets mention a refund?",
        {"intent": "list",
         "filters": [{"field": "issue_summary", "op": "contains", "value": "refund"}]},
    ),
    (
        "What's the sentiment of each customer's message?",
        {"intent": "unsupported",
         "reason": "the dataset has no sentiment column; issue_summary is a short "
                   "free-text label with no sentiment score"},
    ),
]


def build_system_prompt(store) -> str:
    """Assemble the translation prompt from live dataset metadata."""
    examples = "\n\n".join(
        f"Q: {q}\nA: {json.dumps(spec, separators=(',', ':'))}" for q, spec in EXAMPLES
    )

    return f"""You translate questions about a customer support ticket dataset into a \
single JSON query specification. You never answer the question yourself and you \
never see the data -- a Python engine runs your specification and computes the \
result. Your only output is the specification.

DATASET
  {store.row_count} support tickets, {store.min_date.date()} to {store.anchor.date()}.
  Relative dates are resolved against {store.anchor.date()} (the most recent
  ticket), not today's date.

COLUMNS
{_column_lines(store)}

GROUPABLE COLUMNS: {', '.join(GROUPABLE_COLUMNS)}
NUMERIC COLUMNS:   {', '.join(NUMERIC_COLUMNS)}
OPERATORS:         {', '.join(o.value for o in Op)}
METRICS:           {', '.join(m.value for m in Metric)}
DATE PRESETS:      {', '.join(p.value for p in DatePreset)}

SPECIFICATION SHAPE
{SPEC_SHAPE}

{RULES}

EXAMPLES
{examples}

Respond with JSON only."""


def build_repair_prompt(question: str, raw_response: str, error: str) -> str:
    """Second-attempt message: hand the model its own output and the failure."""
    return f"""Your previous specification was rejected.

QUESTION: {question}

YOUR RESPONSE:
{raw_response[:1000]}

ERROR:
{error}

Emit a corrected JSON specification that fixes this error. Use only valid
columns, operators and dataset values. If the question cannot be expressed
within the schema, return {{"intent": "unsupported", "reason": "..."}}.
Respond with JSON only."""
