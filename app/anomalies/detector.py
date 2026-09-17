"""Anomaly detection orchestration.

Runs the four rules in a fixed order, applies filtering, and assembles the
report. Deterministic: the same store and config always produce the same output,
in the same order.
"""

from __future__ import annotations

from app.config import AnomalyConfig, load_anomaly_config
from app.data_store import DataStore
from app.models import (
    AnomalyFilterError,
    AnomalyFlag,
    AnomalyReport,
    Intent,
    QuerySpec,
    Rule,
    Severity,
)
from app.query.executor import describe_conditions, filter_rows
from app.anomalies.rules import RULE_FUNCTIONS, compute_priority_thresholds

SNAPSHOT_NOTE = (
    "This dataset is a static historical snapshot, so every unresolved "
    "high-priority ticket is months old and trips the aging rule. On live data "
    "this rule would flag genuinely stalled work."
)

ASSUMED_SLA_NOTE = (
    "First-response targets are assumed support-desk defaults for this "
    "assessment, not real customer SLA data."
)

# Filtering on the column a rule measures makes that rule circular: narrowing to
# "resolution_time_hrs > 50" and then asking which resolution times are unusual
# computes the IQR threshold from an already-truncated population. These are
# rejected loudly rather than silently ignored.
UNSUPPORTED_FILTER_FIELDS = {
    "resolution_time_hrs": (
        "The resolution-time outlier rule derives its IQR threshold from this "
        "column, so filtering on it would make the threshold circular."
    ),
    "response_time_hrs": (
        "The first-response SLA rule compares this column against its targets, "
        "so filtering on it would pre-select the breaches."
    ),
}

# Population selectors: safe because no rule derives a threshold from them.
SUPPORTED_FILTER_FIELDS = [
    "category", "priority", "status", "agent_id",
    "ticket_id", "issue_summary", "created_at", "customer_rating",
]


def _check_filters(spec: QuerySpec) -> None:
    """Reject filters that would distort the rules they feed."""
    for f in list(spec.filters) + [f for group in spec.or_groups for f in group]:
        if f.field in UNSUPPORTED_FILTER_FIELDS:
            raise AnomalyFilterError(
                f.field, UNSUPPORTED_FILTER_FIELDS[f.field], SUPPORTED_FILTER_FIELDS
            )


def _selection_spec(spec: QuerySpec | None, date_range) -> QuerySpec | None:
    """Build a filter-only spec from either a full spec or a bare date range."""
    if spec is not None:
        if not (spec.filters or spec.or_groups or spec.date_range):
            return None
        return QuerySpec(
            intent=Intent.LIST,
            filters=list(spec.filters),
            or_groups=[list(group) for group in spec.or_groups],
            date_range=spec.date_range,
        )
    if date_range is not None:
        return QuerySpec(intent=Intent.LIST, date_range=date_range)
    return None


def detect(
    store: DataStore,
    config: AnomalyConfig | None = None,
    rule: str | None = None,
    severity: str | None = None,
    limit: int | None = None,
    date_range=None,
    spec: QuerySpec | None = None,
) -> AnomalyReport:
    """Run the enabled rules and return a filtered, ordered report.

    `spec` narrows the population the rules run over, reusing the executor's
    validated filter semantics so there is one definition of what `eq` means.
    Rule 1's IQR thresholds are computed from that narrowed population, which is
    the point: "outliers among Critical tickets" should be judged against
    Critical tickets.

    `rule` and `severity` filter the resulting flags; `limit` truncates them.
    The counts in `summary` describe the full matching set, not the truncated
    page, so a limit never makes the totals lie.
    """
    config = config or load_anomaly_config()
    frame = store.df
    warnings: list[str] = []
    scope: str | None = None

    selection = _selection_spec(spec, date_range)
    if selection is not None:
        _check_filters(selection)
        # Boolean mask only -- the store's frame is never modified.
        frame, applied = filter_rows(store.df, selection, store.anchor)
        scope = describe_conditions(selection, applied) or None

        if len(frame) == 0:
            warnings.append("No tickets matched the filters, so no rules were run.")
        else:
            # Filtering can shrink a priority group below the point where a
            # quartile spread means anything. Rule 1 skips those groups, so say
            # so rather than letting an empty result look like a clean bill.
            outlier_config = config.rule(Rule.RESOLUTION_TIME_OUTLIER.value)
            minimum = int(outlier_config.get("min_group_size", 8))
            covered = set(compute_priority_thresholds(frame, outlier_config))
            resolved = frame[frame["resolution_time_hrs"].notna()]
            skipped = sorted(set(resolved["priority"].unique()) - covered)
            if skipped:
                warnings.append(
                    f"Too few resolved tickets after filtering to compute an "
                    f"outlier threshold for: {', '.join(skipped)} "
                    f"(minimum {minimum}); those priorities were not checked "
                    f"for resolution-time outliers."
                )

    # ---- run rules -------------------------------------------------------
    flags: list[AnomalyFlag] = []
    for rule_id, function in RULE_FUNCTIONS:
        if not config.enabled(rule_id.value):
            continue
        flags.extend(function(frame, config.rule(rule_id.value), store.anchor))

    # Full-set counts, computed before any filtering or truncation.
    summary = {rule_id.value: 0 for rule_id, _ in RULE_FUNCTIONS}
    for flag in flags:
        summary[flag.rule.value] += 1

    severity_summary = {level: 0 for level in config.severity_order}
    for flag in flags:
        severity_summary[flag.severity.value] = severity_summary.get(flag.severity.value, 0) + 1

    # ---- filter ----------------------------------------------------------
    if rule:
        valid = {r.value for r, _ in RULE_FUNCTIONS}
        requested = {r.strip() for r in rule.split(",") if r.strip()}
        unknown = requested - valid
        if unknown:
            warnings.append(
                f"unknown rule(s) ignored: {', '.join(sorted(unknown))}; "
                f"valid rules are {', '.join(sorted(valid))}"
            )
        requested &= valid
        flags = [f for f in flags if f.rule.value in requested] if requested else []

    if severity:
        valid = {s.value for s in Severity}
        requested = {s.strip().lower() for s in severity.split(",") if s.strip()}
        unknown = requested - valid
        if unknown:
            warnings.append(
                f"unknown severity level(s) ignored: {', '.join(sorted(unknown))}; "
                f"valid levels are {', '.join(config.severity_order)}"
            )
        requested &= valid
        flags = [f for f in flags if f.severity.value in requested] if requested else []

    # ---- order -----------------------------------------------------------
    # Severity first, then the biggest breach, then ticket id. The ticket id
    # tie-break is what makes repeated runs byte-identical.
    order = {level: index for index, level in enumerate(config.severity_order)}
    flags.sort(
        key=lambda f: (
            order.get(f.severity.value, 99),
            -(f.observed_value / f.threshold if f.threshold else 0),
            f.ticket_id,
        )
    )

    total = len(flags)
    tickets = len({f.ticket_id for f in flags})

    # Counts for the set that survived filtering, computed before `limit`
    # truncates it. `summary` above stays the full-dataset picture; these
    # describe what the caller actually asked for, so a filtered answer cannot
    # quote unfiltered numbers.
    matched_summary = {rule_id.value: 0 for rule_id, _ in RULE_FUNCTIONS}
    matched_severity = {level: 0 for level in config.severity_order}
    for flag in flags:
        matched_summary[flag.rule.value] += 1
        matched_severity[flag.severity.value] = matched_severity.get(flag.severity.value, 0) + 1

    if limit is not None and limit < total:
        flags = flags[:limit]
        warnings.append(f"showing {limit} of {total} flags")

    # ---- notes -----------------------------------------------------------
    notes = []
    if config.enabled(Rule.UNRESOLVED_HIGH_PRIORITY.value):
        notes.append(SNAPSHOT_NOTE)
    if config.rule(Rule.RESPONSE_SLA_BREACH.value).get("assumed_thresholds", True):
        notes.append(ASSUMED_SLA_NOTE)

    return AnomalyReport(
        flags=flags,
        total_flagged=total,
        returned_count=len(flags),
        tickets_flagged=tickets,
        summary=summary,
        severity_summary=severity_summary,
        matched_summary=matched_summary,
        matched_severity_summary=matched_severity,
        thresholds=describe_thresholds(store, config, frame=frame),
        date_anchor=store.anchor.date().isoformat(),
        scope=scope,
        notes=notes,
        warnings=warnings,
    )


def describe_thresholds(
    store: DataStore, config: AnomalyConfig | None = None, frame=None
) -> dict:
    """The thresholds currently in force, so the UI and API can show the maths.

    `frame` is the filtered population when a filter was requested, so the
    reported IQR thresholds match the ones the rules actually used.
    """
    config = config or load_anomaly_config()
    frame = store.df if frame is None else frame
    outlier = config.rule(Rule.RESOLUTION_TIME_OUTLIER.value)
    sla = config.rule(Rule.RESPONSE_SLA_BREACH.value)
    aging = config.rule(Rule.UNRESOLVED_HIGH_PRIORITY.value)

    return {
        "resolution_time_outlier": {
            "method": f"Q3 + {outlier.get('iqr_multiplier', 1.5)} x IQR, per priority",
            "per_priority": compute_priority_thresholds(frame, outlier),
            "excludes": "rows where resolution precedes first response",
        },
        "unresolved_high_priority": {
            "priorities": aging.get("priorities", []),
            "min_age_hours": aging.get("min_age_hours"),
            "measured_from": store.anchor.date().isoformat(),
        },
        "response_sla_breach": {
            "thresholds_hours": sla.get("thresholds_hours", {}),
            "assumed": sla.get("assumed_thresholds", True),
        },
        "data_integrity_contradiction": {
            "condition": "resolution_time_hrs < response_time_hrs",
        },
    }


def build_answer(report: AnomalyReport, window: str | None = None) -> str:
    """Deterministic one-line summary, scoped to whatever filters were applied."""
    """Deterministic one-line summary.

    Assembled in Python from the counts. There is no second LLM call: the model
    interpreted the question, and the numbers come from the detector.
    """
    described = window or report.scope
    scope = f" where {described}" if described else ""

    if report.total_flagged == 0:
        return f"No anomalies detected{scope}."

    ranked = [
        f"{count} {name.replace('_', ' ')}"
        for name, count in sorted(report.matched_summary.items(), key=lambda kv: -kv[1])
        if count
    ]
    severe = (report.matched_severity_summary.get("critical", 0)
              + report.matched_severity_summary.get("high", 0))

    return (
        f"Found {report.total_flagged} anomal"
        f"{'y' if report.total_flagged == 1 else 'ies'}{scope} across "
        f"{report.tickets_flagged} tickets ({severe} critical or high severity): "
        f"{', '.join(ranked)}."
    )
