"""The four anomaly rules.

Each rule is a pure function of (DataFrame, config, anchor) returning a list of
AnomalyFlag. No LLM, no learned model, no hidden state. Every flag names the
value observed and the threshold it crossed, which is the whole reason these are
rules and not an unsupervised score: a support lead can check the arithmetic.

Nothing here mutates the DataFrame. Rules read it and return records.
"""

from __future__ import annotations

import pandas as pd

from app.models import AnomalyFlag, Rule, Severity

RESOLUTION = "resolution_time_hrs"
RESPONSE = "response_time_hrs"


def _round(value: float, places: int = 2) -> float:
    return float(round(float(value), places))


def _context(row) -> dict:
    """Ticket context attached to every flag so the UI table reads on its own."""
    created = row.get("created_at")
    return {
        "priority": row.get("priority"),
        "status": row.get("status"),
        "category": row.get("category"),
        "agent_id": row.get("agent_id"),
        "created_at": created.isoformat(sep=" ") if pd.notna(created) else None,
    }


# --------------------------------------------------------------------------
# Rule 4 first: the other rules need to know which rows are corrupt.
# --------------------------------------------------------------------------


def corrupt_mask(df: pd.DataFrame) -> pd.Series:
    """Rows where resolution precedes first response.

    Computed separately from the rule itself because Rule 1 needs to exclude
    these rows from its threshold maths, whether or not Rule 4 is enabled.
    """
    return (df[RESOLUTION].notna() & df[RESPONSE].notna() & (df[RESOLUTION] < df[RESPONSE]))


def data_integrity_contradiction(df: pd.DataFrame, config: dict, anchor) -> list[AnomalyFlag]:
    """Rule 4: resolved before the first response. Impossible by definition.

    Zero false positives -- this is arithmetic, not judgement. Flagged rather
    than dropped, because silently discarding rows hides a real data problem.
    """
    severity = Severity(config.get("severity", "high"))
    flags = []

    for _, row in df[corrupt_mask(df)].iterrows():
        resolution, response = float(row[RESOLUTION]), float(row[RESPONSE])
        flags.append(
            AnomalyFlag(
                ticket_id=row["ticket_id"],
                rule=Rule.DATA_INTEGRITY_CONTRADICTION,
                severity=severity,
                metric=RESOLUTION,
                observed_value=_round(resolution),
                # The resolution time cannot legitimately be below the response
                # time, so the response time is the boundary it violated.
                threshold=_round(response),
                explanation=(
                    f"Resolved at {resolution:g}h but first response was at "
                    f"{response:g}h -- resolution precedes the first response, "
                    f"which is impossible. Likely corrupt data."
                ),
                **_context(row),
            )
        )
    return flags


# --------------------------------------------------------------------------
# Rule 1
# --------------------------------------------------------------------------


def compute_priority_thresholds(df: pd.DataFrame, config: dict) -> dict[str, dict]:
    """Q3 + k*IQR per priority, computed on clean resolved rows only.

    Exposed separately so the thresholds can be inspected, returned by the API
    and asserted in tests independently of the flags they produce.
    """
    multiplier = float(config.get("iqr_multiplier", 1.5))
    min_group = int(config.get("min_group_size", 8))

    clean = df[~corrupt_mask(df)]
    thresholds: dict[str, dict] = {}

    for priority, group in clean.groupby("priority", observed=True):
        values = group[RESOLUTION].dropna()
        # A quartile spread over a handful of rows is noise, so small groups are
        # skipped rather than given an unstable threshold.
        if len(values) < min_group:
            continue

        q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
        iqr = q3 - q1
        thresholds[str(priority)] = {
            "q1": _round(q1, 3),
            "q3": _round(q3, 3),
            "iqr": _round(iqr, 3),
            "threshold": _round(q3 + multiplier * iqr, 3),
            "n": int(len(values)),
        }

    return thresholds


def _outlier_severity(ratio: float, ratios: dict) -> Severity:
    if ratio >= float(ratios.get("high", 2.0)):
        return Severity.HIGH
    if ratio >= float(ratios.get("medium", 1.5)):
        return Severity.MEDIUM
    return Severity.LOW


def resolution_time_outlier(df: pd.DataFrame, config: dict, anchor) -> list[AnomalyFlag]:
    """Rule 1: resolution time far above normal for that priority.

    Per-priority because expected duration differs by priority -- a global
    cutoff over-flags Low and under-flags Critical. IQR rather than mean+3sigma
    because the distribution is heavily right-skewed.
    """
    thresholds = compute_priority_thresholds(df, config)
    ratios = config.get("severity_ratios", {})
    corrupt = corrupt_mask(df)
    flags = []

    for priority, stats in thresholds.items():
        threshold = stats["threshold"]
        # Corrupt rows are excluded from the threshold maths above; they are
        # also not re-flagged here, since Rule 4 already reports them and their
        # resolution time is not trustworthy enough to call an outlier.
        candidates = df[
            (df["priority"] == priority) & (df[RESOLUTION] > threshold) & (~corrupt)
        ]

        for _, row in candidates.iterrows():
            observed = float(row[RESOLUTION])
            ratio = observed / threshold if threshold > 0 else float("inf")
            flags.append(
                AnomalyFlag(
                    ticket_id=row["ticket_id"],
                    rule=Rule.RESOLUTION_TIME_OUTLIER,
                    severity=_outlier_severity(ratio, ratios),
                    metric=RESOLUTION,
                    observed_value=_round(observed),
                    threshold=threshold,
                    explanation=(
                        f"Took {observed:g}h to resolve, above the {priority} "
                        f"outlier threshold of {threshold:g}h "
                        f"(Q3 {stats['q3']:g} + 1.5 x IQR {stats['iqr']:g}, "
                        f"n={stats['n']}) -- {ratio:.1f}x the threshold."
                    ),
                    **_context(row),
                )
            )

    return flags


# --------------------------------------------------------------------------
# Rule 2
# --------------------------------------------------------------------------


def _age_severity(age_hours: float, tiers: list[dict]) -> Severity:
    """First tier whose age is met wins; tiers are checked oldest-first."""
    for tier in sorted(tiers, key=lambda t: float(t.get("hours", 0)), reverse=True):
        if age_hours > float(tier.get("hours", 0)):
            return Severity(tier.get("severity", "medium"))
    return Severity.MEDIUM


def unresolved_high_priority(df: pd.DataFrame, config: dict, anchor) -> list[AnomalyFlag]:
    """Rule 2: High/Critical tickets still open past the age threshold.

    Age is measured from the dataset anchor, not wall-clock time, so the rule
    behaves the same in 2026 as it did when the snapshot was taken.

    KNOWN LIMITATION: on this static Q1-2024 snapshot every unresolved
    High/Critical ticket is months old, so all 80 trip the rule. It is tiered by
    age and sorted oldest-first so the output is still usable as a backlog view,
    but on live data this would be a genuine alert rather than a listing.
    """
    priorities = config.get("priorities", ["Critical", "High"])
    min_age = float(config.get("min_age_hours", 24))
    tiers = config.get("severity_tiers", [])

    ages = (anchor - df["created_at"]).dt.total_seconds() / 3600.0
    candidates = df[
        df["priority"].isin(priorities) & (df["status"] != "Resolved") & (ages > min_age)
    ]

    flags = []
    for index, row in candidates.iterrows():
        age = float(ages.loc[index])
        days = age / 24.0
        flags.append(
            AnomalyFlag(
                ticket_id=row["ticket_id"],
                rule=Rule.UNRESOLVED_HIGH_PRIORITY,
                severity=_age_severity(age, tiers),
                metric="age_hours",
                observed_value=_round(age, 1),
                threshold=_round(min_age, 1),
                explanation=(
                    f"{row['priority']} ticket still {row['status']} after "
                    f"{age:.1f}h ({days:.1f} days), past the {min_age:g}h "
                    f"threshold. Age measured from the dataset anchor "
                    f"{anchor.date()}."
                ),
                **_context(row),
            )
        )

    # Oldest first: the most overdue work is what a support lead needs on top.
    flags.sort(key=lambda f: f.observed_value, reverse=True)
    return flags


# --------------------------------------------------------------------------
# Rule 3
# --------------------------------------------------------------------------


def response_sla_breach(df: pd.DataFrame, config: dict, anchor) -> list[AnomalyFlag]:
    """Rule 3: first response slower than the target for that priority.

    The one rule with full coverage: response_time_hrs has no nulls, so it
    applies to all 500 tickets rather than only the resolved ones.

    ASSUMED THRESHOLDS -- reasonable support-desk defaults invented for this
    assessment, not real customer SLA data.
    """
    thresholds = config.get("thresholds_hours", {})
    severities = config.get("severities", {})
    assumed = config.get("assumed_thresholds", True)
    note = " (assumed threshold, not a real customer SLA)" if assumed else ""

    flags = []
    for priority, limit in thresholds.items():
        limit = float(limit)
        candidates = df[(df["priority"] == priority) & (df[RESPONSE] > limit)]

        for _, row in candidates.iterrows():
            observed = float(row[RESPONSE])
            flags.append(
                AnomalyFlag(
                    ticket_id=row["ticket_id"],
                    rule=Rule.RESPONSE_SLA_BREACH,
                    severity=Severity(severities.get(priority, "medium")),
                    metric=RESPONSE,
                    observed_value=_round(observed),
                    threshold=_round(limit),
                    explanation=(
                        f"First response took {observed:g}h, past the "
                        f"{limit:g}h target for {priority} tickets{note}."
                    ),
                    **_context(row),
                )
            )

    return flags


# Fixed order, so a full report is byte-identical across runs.
RULE_FUNCTIONS = [
    (Rule.RESOLUTION_TIME_OUTLIER, resolution_time_outlier),
    (Rule.UNRESOLVED_HIGH_PRIORITY, unresolved_high_priority),
    (Rule.RESPONSE_SLA_BREACH, response_sla_breach),
    (Rule.DATA_INTEGRITY_CONTRADICTION, data_integrity_contradiction),
]
