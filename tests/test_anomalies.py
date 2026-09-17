"""Anomaly rules.

Expected values are hand-computed from the CSV or built into small synthetic
frames with known answers, rather than recomputed with the implementation's own
logic. A test that calls compute_priority_thresholds to build its expectation
would pass even if the formula were wrong.

Ground truth for the real dataset (verified independently with pandas):
    corrupt rows (resolution < response)      28
    resolution outliers (clean, per-priority) 18
    unresolved High/Critical older than 24h   80
    first-response SLA breaches              160
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.anomalies import rules
from app.anomalies.detector import build_answer, detect
from app.config import AnomalyConfig, load_anomaly_config
from app.models import (
    AnomalyFilterError,
    DatePreset,
    DateRange,
    Filter,
    Intent,
    Op,
    QuerySpec,
    Rule,
    Severity,
)


@pytest.fixture(scope="module")
def config():
    return load_anomaly_config()


@pytest.fixture
def frame(store):
    return store.df


def make_frame(rows: list[dict]) -> pd.DataFrame:
    """Small synthetic frame with the real column set."""
    base = {
        "ticket_id": "TKT-000", "created_at": pd.Timestamp("2024-03-01 09:00"),
        "category": "Billing", "priority": "Medium", "status": "Resolved",
        "response_time_hrs": 1.0, "resolution_time_hrs": 5.0,
        "agent_id": "AGT-01", "customer_rating": 4, "issue_summary": "x",
    }
    return pd.DataFrame([{**base, **row} for row in rows])


ANCHOR = pd.Timestamp("2024-03-30 18:06")


# ==================================================== Rule 4: data integrity


def test_integrity_rule_finds_the_known_28(frame, config):
    flags = rules.data_integrity_contradiction(
        frame, config.rule(Rule.DATA_INTEGRITY_CONTRADICTION.value), ANCHOR)
    assert len(flags) == 28


def test_integrity_flag_reports_both_values(config):
    df = make_frame([{"ticket_id": "TKT-A", "response_time_hrs": 4.8,
                      "resolution_time_hrs": 2.1}])
    flag = rules.data_integrity_contradiction(
        df, config.rule(Rule.DATA_INTEGRITY_CONTRADICTION.value), ANCHOR)[0]

    assert flag.observed_value == 2.1
    assert flag.threshold == 4.8
    assert flag.severity is Severity.HIGH
    assert "2.1h" in flag.explanation and "4.8h" in flag.explanation


def test_integrity_equal_times_are_not_flagged(config):
    """Strictly less-than: equal values are odd but not impossible."""
    df = make_frame([{"response_time_hrs": 3.0, "resolution_time_hrs": 3.0}])
    assert rules.data_integrity_contradiction(
        df, config.rule(Rule.DATA_INTEGRITY_CONTRADICTION.value), ANCHOR) == []


def test_integrity_ignores_unresolved_tickets(config):
    df = make_frame([{"status": "Open", "resolution_time_hrs": None,
                      "response_time_hrs": 4.0}])
    assert rules.data_integrity_contradiction(
        df, config.rule(Rule.DATA_INTEGRITY_CONTRADICTION.value), ANCHOR) == []


def test_corrupt_mask_matches_the_rule(frame):
    assert int(rules.corrupt_mask(frame).sum()) == 28


# ================================================ Rule 1: resolution outlier


def test_priority_thresholds_match_hand_computed_values(frame, config):
    """Q3 + 1.5*IQR per priority, computed by hand from the clean rows."""
    thresholds = rules.compute_priority_thresholds(
        frame, config.rule(Rule.RESOLUTION_TIME_OUTLIER.value))

    expected = {
        "Critical": (5.8, 2.075, 8.913, 14),
        "High": (10.4, 5.6, 18.8, 72),
        "Medium": (20.35, 11.425, 37.488, 112),
        "Low": (38.5, 24.0, 74.5, 101),
    }
    for priority, (q3, iqr, threshold, n) in expected.items():
        actual = thresholds[priority]
        assert actual["q3"] == pytest.approx(q3, abs=0.01)
        assert actual["iqr"] == pytest.approx(iqr, abs=0.01)
        assert actual["threshold"] == pytest.approx(threshold, abs=0.01)
        assert actual["n"] == n


def test_thresholds_differ_by_priority(frame, config):
    """The reason for per-priority: a global cutoff would be wrong for both."""
    thresholds = rules.compute_priority_thresholds(
        frame, config.rule(Rule.RESOLUTION_TIME_OUTLIER.value))
    assert thresholds["Critical"]["threshold"] < 10
    assert thresholds["Low"]["threshold"] > 70


def test_corrupt_rows_are_excluded_from_threshold_maths(frame, config):
    """Including them shifts the Critical threshold from 8.913 to 9.775."""
    rule_config = config.rule(Rule.RESOLUTION_TIME_OUTLIER.value)
    clean = rules.compute_priority_thresholds(frame, rule_config)

    # Recompute the naive way, without the exclusion.
    values = frame[frame["priority"] == "Critical"]["resolution_time_hrs"].dropna()
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    naive = q3 + 1.5 * (q3 - q1)

    assert clean["Critical"]["threshold"] == pytest.approx(8.913, abs=0.01)
    assert naive == pytest.approx(9.775, abs=0.01)
    assert clean["Critical"]["threshold"] != pytest.approx(naive, abs=0.01)


def test_outlier_rule_finds_the_known_18(frame, config):
    flags = rules.resolution_time_outlier(
        frame, config.rule(Rule.RESOLUTION_TIME_OUTLIER.value), ANCHOR)
    assert len(flags) == 18


def test_corrupt_rows_are_not_themselves_flagged_as_outliers(frame, config):
    outliers = {f.ticket_id for f in rules.resolution_time_outlier(
        frame, config.rule(Rule.RESOLUTION_TIME_OUTLIER.value), ANCHOR)}
    corrupt = set(frame[rules.corrupt_mask(frame)]["ticket_id"])
    assert outliers & corrupt == set()


def test_outlier_threshold_formula_on_a_known_group(config):
    """Values 1..10: Q1=3.25, Q3=7.75, IQR=4.5, threshold = 7.75 + 6.75 = 14.5."""
    rows = [{"ticket_id": f"TKT-{i:03}", "priority": "Medium",
             "resolution_time_hrs": float(i), "response_time_hrs": 0.1}
            for i in range(1, 11)]
    thresholds = rules.compute_priority_thresholds(
        make_frame(rows), {"iqr_multiplier": 1.5, "min_group_size": 5})
    assert thresholds["Medium"]["q1"] == pytest.approx(3.25)
    assert thresholds["Medium"]["q3"] == pytest.approx(7.75)
    assert thresholds["Medium"]["threshold"] == pytest.approx(14.5)


def _constant_group(extra: list[dict]) -> pd.DataFrame:
    """20 identical rows plus candidates.

    A constant bulk keeps both quartiles pinned at 10.0 no matter what is added
    at the top, so the threshold is exactly 10.0 by hand and inserting the
    candidate row cannot move the boundary being tested.
    """
    rows = [{"ticket_id": f"TKT-B{i:03}", "priority": "Medium",
             "resolution_time_hrs": 10.0, "response_time_hrs": 0.1}
            for i in range(20)]
    return make_frame(rows + extra)


def test_outlier_boundary_is_strictly_above(config):
    """A value exactly at the threshold is not an outlier."""
    rule_config = {"iqr_multiplier": 1.5, "min_group_size": 5, "severity_ratios": {}}

    at = _constant_group([{"ticket_id": "TKT-AT", "priority": "Medium",
                           "resolution_time_hrs": 10.0, "response_time_hrs": 0.1}])
    assert rules.compute_priority_thresholds(at, rule_config)["Medium"]["threshold"] == 10.0
    assert rules.resolution_time_outlier(at, rule_config, ANCHOR) == []

    over = _constant_group([{"ticket_id": "TKT-OVER", "priority": "Medium",
                             "resolution_time_hrs": 10.5, "response_time_hrs": 0.1}])
    flags = rules.resolution_time_outlier(over, rule_config, ANCHOR)
    assert [f.ticket_id for f in flags] == ["TKT-OVER"]
    assert flags[0].threshold == 10.0
    assert flags[0].observed_value == 10.5


def test_small_groups_are_skipped(config):
    """Quartiles over three rows are noise, not a threshold."""
    rows = [{"ticket_id": f"TKT-{i}", "priority": "Critical",
             "resolution_time_hrs": float(i)} for i in range(3)]
    thresholds = rules.compute_priority_thresholds(
        make_frame(rows), {"iqr_multiplier": 1.5, "min_group_size": 8})
    assert thresholds == {}


def test_outlier_severity_scales_with_overshoot(config):
    """Threshold is 10.0; candidates sit at 1.05x, 1.5x and 2.0x exactly."""
    rule_config = {"iqr_multiplier": 1.5, "min_group_size": 5,
                   "severity_ratios": {"high": 2.0, "medium": 1.5}}
    df = _constant_group([
        {"ticket_id": "TKT-LOW", "priority": "Medium", "resolution_time_hrs": 10.5,
         "response_time_hrs": 0.1},
        {"ticket_id": "TKT-MED", "priority": "Medium", "resolution_time_hrs": 15.0,
         "response_time_hrs": 0.1},
        {"ticket_id": "TKT-HIGH", "priority": "Medium", "resolution_time_hrs": 20.0,
         "response_time_hrs": 0.1},
    ])
    assert rules.compute_priority_thresholds(df, rule_config)["Medium"]["threshold"] == 10.0
    by_id = {f.ticket_id: f.severity for f in
             rules.resolution_time_outlier(df, rule_config, ANCHOR)}
    assert by_id["TKT-LOW"] is Severity.LOW
    assert by_id["TKT-MED"] is Severity.MEDIUM
    assert by_id["TKT-HIGH"] is Severity.HIGH


def test_outlier_explanation_shows_the_maths(frame, config):
    flag = rules.resolution_time_outlier(
        frame, config.rule(Rule.RESOLUTION_TIME_OUTLIER.value), ANCHOR)[0]
    assert "IQR" in flag.explanation and "threshold" in flag.explanation
    assert flag.observed_value > flag.threshold


# =========================================== Rule 2: unresolved high priority


def test_aging_rule_finds_the_known_80(frame, config):
    flags = rules.unresolved_high_priority(
        frame, config.rule(Rule.UNRESOLVED_HIGH_PRIORITY.value), ANCHOR)
    assert len(flags) == 80


def test_aging_rule_only_covers_high_and_critical(frame, config):
    flags = rules.unresolved_high_priority(
        frame, config.rule(Rule.UNRESOLVED_HIGH_PRIORITY.value), ANCHOR)
    assert {f.priority for f in flags} <= {"Critical", "High"}
    assert all(f.status != "Resolved" for f in flags)


def test_aging_rule_uses_the_dataset_anchor_not_today(frame, config):
    """Wall-clock ages would be ~2.5 years; anchored ages max out near 88 days."""
    flags = rules.unresolved_high_priority(
        frame, config.rule(Rule.UNRESOLVED_HIGH_PRIORITY.value), ANCHOR)
    assert max(f.observed_value for f in flags) < 24 * 100
    assert "2024-03-30" in flags[0].explanation


def test_aging_boundary_is_strictly_above(config):
    rule_config = {"priorities": ["Critical"], "min_age_hours": 24,
                   "severity_tiers": [{"hours": 24, "severity": "medium"}]}
    df = make_frame([
        {"ticket_id": "TKT-AT", "priority": "Critical", "status": "Open",
         "created_at": ANCHOR - pd.Timedelta(hours=24)},
        {"ticket_id": "TKT-OVER", "priority": "Critical", "status": "Open",
         "created_at": ANCHOR - pd.Timedelta(hours=24, minutes=1)},
        {"ticket_id": "TKT-UNDER", "priority": "Critical", "status": "Open",
         "created_at": ANCHOR - pd.Timedelta(hours=23)},
    ])
    found = {f.ticket_id for f in rules.unresolved_high_priority(df, rule_config, ANCHOR)}
    assert found == {"TKT-OVER"}


def test_aging_severity_tiers(config):
    rule_config = config.rule(Rule.UNRESOLVED_HIGH_PRIORITY.value)
    df = make_frame([
        {"ticket_id": "TKT-M", "priority": "Critical", "status": "Open",
         "created_at": ANCHOR - pd.Timedelta(hours=30)},    # 24-72  -> medium
        {"ticket_id": "TKT-H", "priority": "Critical", "status": "Open",
         "created_at": ANCHOR - pd.Timedelta(hours=100)},   # 72-168 -> high
        {"ticket_id": "TKT-C", "priority": "Critical", "status": "Open",
         "created_at": ANCHOR - pd.Timedelta(hours=200)},   # >168   -> critical
    ])
    by_id = {f.ticket_id: f.severity for f in
             rules.unresolved_high_priority(df, rule_config, ANCHOR)}
    assert by_id["TKT-M"] is Severity.MEDIUM
    assert by_id["TKT-H"] is Severity.HIGH
    assert by_id["TKT-C"] is Severity.CRITICAL


def test_aging_tier_counts_on_real_data(frame, config):
    """Independently computed: 5 in 24-72h, 4 in 72-168h, 71 beyond a week."""
    flags = rules.unresolved_high_priority(
        frame, config.rule(Rule.UNRESOLVED_HIGH_PRIORITY.value), ANCHOR)
    counts = {"medium": 0, "high": 0, "critical": 0}
    for flag in flags:
        counts[flag.severity.value] += 1
    assert counts == {"medium": 5, "high": 4, "critical": 71}


def test_aging_flags_are_sorted_oldest_first(frame, config):
    flags = rules.unresolved_high_priority(
        frame, config.rule(Rule.UNRESOLVED_HIGH_PRIORITY.value), ANCHOR)
    ages = [f.observed_value for f in flags]
    assert ages == sorted(ages, reverse=True)


def test_aging_threshold_is_configurable(frame):
    """Raising the threshold past the oldest ticket empties the rule."""
    tight = {"priorities": ["Critical", "High"], "min_age_hours": 100000,
             "severity_tiers": [{"hours": 24, "severity": "medium"}]}
    assert rules.unresolved_high_priority(frame, tight, ANCHOR) == []


def test_aging_rule_preserves_the_snapshot_limitation(frame, config):
    """All 80 unresolved High/Critical tickets trip it -- documented, not hidden."""
    unresolved = frame[
        frame["priority"].isin(["Critical", "High"]) & (frame["status"] != "Resolved")]
    flags = rules.unresolved_high_priority(
        frame, config.rule(Rule.UNRESOLVED_HIGH_PRIORITY.value), ANCHOR)
    assert len(flags) == len(unresolved) == 80


# ================================================ Rule 3: response SLA breach


def test_sla_rule_finds_the_known_160(frame, config):
    flags = rules.response_sla_breach(
        frame, config.rule(Rule.RESPONSE_SLA_BREACH.value), ANCHOR)
    assert len(flags) == 160


@pytest.mark.parametrize(
    "priority,expected", [("Critical", 46), ("High", 77), ("Medium", 37)])
def test_sla_breaches_per_priority(frame, config, priority, expected):
    """Hand-computed per priority: Critical >1h, High >2h, Medium >4h."""
    flags = rules.response_sla_breach(
        frame, config.rule(Rule.RESPONSE_SLA_BREACH.value), ANCHOR)
    assert len([f for f in flags if f.priority == priority]) == expected


def test_low_priority_has_no_sla(frame, config):
    flags = rules.response_sla_breach(
        frame, config.rule(Rule.RESPONSE_SLA_BREACH.value), ANCHOR)
    assert not [f for f in flags if f.priority == "Low"]


@pytest.mark.parametrize(
    "priority,limit", [("Critical", 1.0), ("High", 2.0), ("Medium", 4.0)])
def test_sla_boundary_is_strictly_above(config, priority, limit):
    df = make_frame([
        {"ticket_id": "TKT-AT", "priority": priority, "response_time_hrs": limit},
        {"ticket_id": "TKT-OVER", "priority": priority, "response_time_hrs": limit + 0.1},
        {"ticket_id": "TKT-UNDER", "priority": priority, "response_time_hrs": limit - 0.1},
    ])
    found = {f.ticket_id for f in rules.response_sla_breach(
        df, config.rule(Rule.RESPONSE_SLA_BREACH.value), ANCHOR)}
    assert found == {"TKT-OVER"}


def test_sla_severity_by_priority(frame, config):
    flags = rules.response_sla_breach(
        frame, config.rule(Rule.RESPONSE_SLA_BREACH.value), ANCHOR)
    by_priority = {f.priority: f.severity for f in flags}
    assert by_priority["Critical"] is Severity.HIGH
    assert by_priority["High"] is Severity.MEDIUM
    assert by_priority["Medium"] is Severity.LOW


def test_sla_explanation_marks_thresholds_as_assumed(frame, config):
    """Honesty: these are invented defaults, not real customer SLA data."""
    flag = rules.response_sla_breach(
        frame, config.rule(Rule.RESPONSE_SLA_BREACH.value), ANCHOR)[0]
    assert "assumed" in flag.explanation.lower()


def test_sla_thresholds_are_configurable(frame):
    loose = {"thresholds_hours": {"Critical": 100}, "severities": {"Critical": "low"},
             "assumed_thresholds": True}
    assert rules.response_sla_breach(frame, loose, ANCHOR) == []


def test_sla_rule_covers_all_tickets_not_just_resolved(frame, config):
    """response_time_hrs has no nulls, so this rule has full coverage."""
    flags = rules.response_sla_breach(
        frame, config.rule(Rule.RESPONSE_SLA_BREACH.value), ANCHOR)
    assert {f.status for f in flags} == {"Open", "Resolved", "Escalated"}


# ==================================================== structured output shape


def test_every_flag_has_the_required_fields(store, config):
    for flag in detect(store, config).flags:
        assert flag.ticket_id and flag.rule and flag.severity
        assert flag.metric
        assert isinstance(flag.observed_value, float)
        assert isinstance(flag.threshold, float)
        assert len(flag.explanation) > 20


def test_flags_are_json_serialisable(store, config):
    report = detect(store, config, limit=5)
    payload = report.model_dump_json()
    assert "NaN" not in payload


def test_rule_names_are_stable_identifiers(store, config):
    """The API, UI and tests all filter on these exact strings."""
    names = {f.rule.value for f in detect(store, config).flags}
    assert names == {
        "resolution_time_outlier", "unresolved_high_priority",
        "response_sla_breach", "data_integrity_contradiction",
    }


def test_severity_values_are_from_the_fixed_set(store, config):
    severities = {f.severity.value for f in detect(store, config).flags}
    assert severities <= {"critical", "high", "medium", "low"}


# ======================================================== detector behaviour


def test_full_report_counts_match_ground_truth(store, config):
    report = detect(store, config, limit=500)
    assert report.summary == {
        "resolution_time_outlier": 18,
        "unresolved_high_priority": 80,
        "response_sla_breach": 160,
        "data_integrity_contradiction": 28,
    }
    assert report.total_flagged == 286
    assert report.tickets_flagged == 206


def test_report_includes_thresholds_and_anchor(store, config):
    report = detect(store, config)
    assert report.date_anchor == "2024-03-30"
    assert report.thresholds["response_sla_breach"]["thresholds_hours"]["Critical"] == 1
    assert "Critical" in report.thresholds["resolution_time_outlier"]["per_priority"]


def test_report_notes_document_the_known_limitations(store, config):
    notes = " ".join(detect(store, config).notes).lower()
    assert "snapshot" in notes
    assert "assumed" in notes


# ------------------------------------------------------------------ filtering


@pytest.mark.parametrize(
    "rule_name,expected",
    [("resolution_time_outlier", 18), ("unresolved_high_priority", 80),
     ("response_sla_breach", 160), ("data_integrity_contradiction", 28)],
)
def test_rule_filter(store, config, rule_name, expected):
    report = detect(store, config, rule=rule_name, limit=500)
    assert report.total_flagged == expected
    assert {f.rule.value for f in report.flags} == {rule_name}


def test_multiple_rule_filter(store, config):
    report = detect(store, config,
                    rule="response_sla_breach,data_integrity_contradiction", limit=500)
    assert report.total_flagged == 160 + 28


def test_severity_filter(store, config):
    report = detect(store, config, severity="critical", limit=500)
    assert report.total_flagged == 71
    assert {f.severity.value for f in report.flags} == {"critical"}


def test_combined_rule_and_severity_filter(store, config):
    report = detect(store, config, rule="resolution_time_outlier",
                    severity="high", limit=500)
    assert all(f.rule.value == "resolution_time_outlier" and f.severity.value == "high"
               for f in report.flags)


def test_unknown_rule_returns_nothing_and_warns(store, config):
    report = detect(store, config, rule="not_a_rule")
    assert report.total_flagged == 0
    assert any("unknown rule" in w for w in report.warnings)


def test_unknown_severity_returns_nothing_and_warns(store, config):
    report = detect(store, config, severity="catastrophic")
    assert report.total_flagged == 0
    assert any("unknown severity" in w for w in report.warnings)


def test_limit_truncates_without_distorting_counts(store, config):
    report = detect(store, config, limit=10)
    assert report.returned_count == 10
    assert report.total_flagged == 286          # full count preserved
    assert report.summary["response_sla_breach"] == 160
    assert any("showing 10 of 286" in w for w in report.warnings)


def test_filtered_summary_does_not_quote_unfiltered_numbers(store, config):
    """A filtered answer must not report the whole dataset's counts."""
    report = detect(store, config, rule="data_integrity_contradiction", limit=500)
    assert report.matched_summary["data_integrity_contradiction"] == 28
    assert report.matched_summary["response_sla_breach"] == 0
    assert "160" not in build_answer(report)


def test_empty_result_is_clean(store, config):
    report = detect(store, config, rule="resolution_time_outlier", severity="critical")
    assert report.flags == []
    assert report.total_flagged == 0
    assert build_answer(report) == "No anomalies detected."


def test_disabled_rules_produce_no_flags(store):
    disabled = AnomalyConfig(
        rules={name: {**body, "enabled": False}
               for name, body in load_anomaly_config().rules.items()},
        severity_order=["critical", "high", "medium", "low"],
    )
    assert detect(store, disabled).total_flagged == 0


# ------------------------------------------------------------- ordering etc.


def test_flags_are_ordered_by_severity(store, config):
    order = ["critical", "high", "medium", "low"]
    positions = [order.index(f.severity.value) for f in detect(store, config, limit=500).flags]
    assert positions == sorted(positions)


def test_detection_is_deterministic(store, config):
    first = detect(store, config, limit=500)
    second = detect(store, config, limit=500)
    assert first.model_dump() == second.model_dump()


def test_detection_does_not_mutate_the_dataframe(store, config):
    before = store.df.copy(deep=True)
    detect(store, config, limit=500)
    pd.testing.assert_frame_equal(store.df, before)
    assert len(store.df) == 500


def test_no_llm_is_involved(store, config):
    """The detector takes a store and a config -- there is nowhere to pass a model."""
    import inspect

    assert "client" not in inspect.signature(detect).parameters
    assert "translator" not in inspect.signature(detect).parameters
    # No model, no network: the rules module imports only pandas and our models.
    imported = {line.split()[1] for line in inspect.getsource(rules).splitlines()
                if line.startswith("import ") or line.startswith("from ")}
    assert not any(name.startswith(("app.llm", "requests", "openai")) for name in imported)


# ----------------------------------------------------------------- windowing


def test_date_window_narrows_detection(store, config):
    from app.models import DatePreset, DateRange

    windowed = detect(store, config, date_range=DateRange(preset=DatePreset.THIS_MONTH),
                      limit=500)
    full = detect(store, config, limit=500)
    assert windowed.total_flagged < full.total_flagged
    assert windowed.total_flagged > 0


def test_window_does_not_mutate_the_store(store, config):
    from app.models import DatePreset, DateRange

    detect(store, config, date_range=DateRange(preset=DatePreset.TODAY))
    assert len(store.df) == 500


# ========================================================== filtered detection
# Expected values below were computed independently with pandas against the real
# CSV, not by calling the detector.
#
#   Critical only          rule1=3  rule2=31 rule3=46 rule4=10  total=90
#   this_week (Mar 25-31)  rule1=1  rule2=6  rule3=14 rule4=3   total=24
#   High + this_week       rule1=0  rule2=4  rule3=4  rule4=2   total=10
#   Technical only         rule1=7  rule2=22 rule3=47 rule4=9   total=85


def anomaly_spec(**kwargs) -> QuerySpec:
    return QuerySpec(intent=Intent.ANOMALY, **kwargs)


def test_critical_only_filter(store, config):
    spec = anomaly_spec(filters=[Filter(field="priority", op=Op.EQ, value="Critical")])
    report = detect(store, config, spec=spec, limit=500)

    assert report.total_flagged == 90
    assert report.matched_summary == {
        "resolution_time_outlier": 3,
        "unresolved_high_priority": 31,
        "response_sla_breach": 46,
        "data_integrity_contradiction": 10,
    }
    assert all(f.priority == "Critical" for f in report.flags)


def test_high_priority_this_week(store, config):
    spec = anomaly_spec(
        filters=[Filter(field="priority", op=Op.EQ, value="High")],
        date_range=DateRange(preset=DatePreset.THIS_WEEK),
    )
    report = detect(store, config, spec=spec, limit=500)

    assert report.total_flagged == 10
    assert all(f.priority == "High" for f in report.flags)
    assert "priority is High" in report.scope and "this week" in report.scope


def test_date_only_filter(store, config):
    report = detect(store, config,
                    spec=anomaly_spec(date_range=DateRange(preset=DatePreset.THIS_WEEK)),
                    limit=500)
    assert report.total_flagged == 24


def test_category_filter(store, config):
    spec = anomaly_spec(filters=[Filter(field="category", op=Op.EQ, value="Technical")])
    report = detect(store, config, spec=spec, limit=500)
    assert report.total_flagged == 85
    assert all(f.category == "Technical" for f in report.flags)


def test_filter_values_are_normalised_like_any_query(store, config):
    """'crit' resolves to 'Critical' through the same normalisation path."""
    from app.data_store import normalize_spec

    spec = normalize_spec(
        anomaly_spec(filters=[Filter(field="priority", op=Op.EQ, value="crit")]), store)
    assert detect(store, config, spec=spec, limit=500).total_flagged == 90


def test_or_groups_are_honoured(store, config):
    spec = anomaly_spec(or_groups=[[
        Filter(field="priority", op=Op.EQ, value="Critical"),
        Filter(field="priority", op=Op.EQ, value="High"),
    ]])
    report = detect(store, config, spec=spec, limit=500)
    assert {f.priority for f in report.flags} == {"Critical", "High"}


# ------------------------------------------------- Rule 1 on the filtered set


def test_rule1_threshold_uses_the_filtered_population(store, config):
    """Outliers among Critical tickets are judged against Critical tickets."""
    spec = anomaly_spec(filters=[Filter(field="priority", op=Op.EQ, value="Critical")])
    report = detect(store, config, spec=spec, limit=500)

    per_priority = report.thresholds["resolution_time_outlier"]["per_priority"]
    assert set(per_priority) == {"Critical"}
    assert per_priority["Critical"]["threshold"] == pytest.approx(8.913, abs=0.01)


def test_reported_thresholds_match_the_rules_that_ran(store, config):
    spec = anomaly_spec(filters=[Filter(field="category", op=Op.EQ, value="Technical")])
    report = detect(store, config, spec=spec, limit=500)

    per_priority = report.thresholds["resolution_time_outlier"]["per_priority"]
    for flag in report.flags:
        if flag.rule is Rule.RESOLUTION_TIME_OUTLIER:
            assert flag.threshold == per_priority[flag.priority]["threshold"]


def test_small_group_after_filtering_warns(store, config):
    """A silent empty result would look like a clean bill of health."""
    spec = anomaly_spec(
        filters=[Filter(field="priority", op=Op.EQ, value="Critical")],
        date_range=DateRange(preset=DatePreset.THIS_WEEK),
    )
    report = detect(store, config, spec=spec, limit=500)
    assert any("Too few resolved tickets" in w for w in report.warnings)


# ------------------------------------------------------- unsupported filters


@pytest.mark.parametrize("field", ["resolution_time_hrs", "response_time_hrs"])
def test_filtering_on_a_measured_column_is_rejected(store, config, field):
    spec = anomaly_spec(filters=[Filter(field=field, op=Op.GT, value=10)])
    with pytest.raises(AnomalyFilterError) as exc:
        detect(store, config, spec=spec)
    assert exc.value.field == field
    assert "priority" in exc.value.allowed


def test_unsupported_filter_inside_an_or_group_is_caught(store, config):
    spec = anomaly_spec(or_groups=[[
        Filter(field="priority", op=Op.EQ, value="Critical"),
        Filter(field="resolution_time_hrs", op=Op.GT, value=10),
    ]])
    with pytest.raises(AnomalyFilterError):
        detect(store, config, spec=spec)


def test_unsupported_filter_error_is_structured(store, config):
    spec = anomaly_spec(filters=[Filter(field="resolution_time_hrs", op=Op.GT, value=10)])
    with pytest.raises(AnomalyFilterError) as exc:
        detect(store, config, spec=spec)
    payload = exc.value.to_dict()
    assert payload["error"] == "unsupported_anomaly_filter"
    assert payload["allowed_filter_fields"]


def test_unsupported_filter_is_not_silently_ignored(store, config):
    """The failure mode this fix exists to prevent."""
    from app.models import UnsupportedQueryError

    spec = anomaly_spec(filters=[Filter(field="response_time_hrs", op=Op.GT, value=1)])
    with pytest.raises(UnsupportedQueryError):
        detect(store, config, spec=spec)


# ------------------------------------------------------------- no regression


def test_no_filter_still_gives_the_full_dataset_result(store, config):
    report = detect(store, config, limit=500)
    assert report.total_flagged == 286
    assert report.tickets_flagged == 206
    assert report.scope is None
    assert report.summary == {
        "resolution_time_outlier": 18,
        "unresolved_high_priority": 80,
        "response_sla_breach": 160,
        "data_integrity_contradiction": 28,
    }


def test_empty_spec_behaves_like_no_filter(store, config):
    assert detect(store, config, spec=anomaly_spec(), limit=500).total_flagged == 286


def test_legacy_date_range_argument_still_works(store, config):
    by_range = detect(store, config, date_range=DateRange(preset=DatePreset.THIS_WEEK),
                      limit=500)
    by_spec = detect(store, config,
                     spec=anomaly_spec(date_range=DateRange(preset=DatePreset.THIS_WEEK)),
                     limit=500)
    assert by_range.total_flagged == by_spec.total_flagged == 24


def test_filter_matching_nothing_is_clean(store, config):
    # A window entirely outside the dataset (which ends 2024-03-30).
    spec = anomaly_spec(date_range=DateRange(start="2023-01-01", end="2023-01-31"))
    report = detect(store, config, spec=spec, limit=500)
    assert report.total_flagged == 0
    assert report.flags == []
    assert any("No tickets matched" in w for w in report.warnings)
    assert build_answer(report).startswith("No anomalies detected")


def test_filtered_detection_is_deterministic(store, config):
    spec = anomaly_spec(filters=[Filter(field="priority", op=Op.EQ, value="Critical")])
    assert (detect(store, config, spec=spec, limit=500).model_dump()
            == detect(store, config, spec=spec, limit=500).model_dump())


def test_filtered_detection_does_not_mutate_the_store(store, config):
    before = store.df.copy(deep=True)
    detect(store, config,
           spec=anomaly_spec(filters=[Filter(field="priority", op=Op.EQ, value="Critical")]),
           limit=500)
    pd.testing.assert_frame_equal(store.df, before)
