"""Runtime configuration, read from the environment.

Deliberately plain os.getenv rather than a settings framework: there are eight
values here and the evaluator needs to run this from a .env file with no
surprises.
"""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional; .env is a convenience, not a requirement
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    # Provider selection. `provider` is tried first; `fallback_provider` is used
    # only if the first is unreachable. Set fallback to "none" to disable.
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "groq").lower())
    fallback_provider: str = field(
        default_factory=lambda: os.getenv("LLM_FALLBACK_PROVIDER", "ollama").lower()
    )

    groq_api_key: str | None = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    # Verified against console.groq.com/docs/models (September 2026).
    # gpt-oss-120b is a Groq *production* model reachable on the no-credit-card
    # free tier. The Llama models this previously defaulted to
    # (llama-3.3-70b-versatile, llama-3.1-8b-instant) are now Enterprise-only,
    # so they would fail for an evaluator without a sales contract.
    # openai/gpt-oss-20b is the faster, lighter alternative.
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    )
    groq_base_url: str = field(
        default_factory=lambda: os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    )

    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.1:8b"))
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )

    # Translation is short structured extraction, not generation.
    temperature: float = 0.0
    max_tokens: int = field(default_factory=lambda: _int("LLM_MAX_TOKENS", 800))
    timeout_seconds: int = field(default_factory=lambda: _int("LLM_TIMEOUT_SECONDS", 30))

    # One repair attempt. Not an open-ended agent loop: if the model cannot
    # produce a valid spec with the validation error handed back to it, more
    # turns mostly buy latency.
    max_repair_attempts: int = 1

    # Guard against prompt-stuffing through the question field.
    max_question_length: int = field(default_factory=lambda: _int("MAX_QUESTION_LENGTH", 500))


def get_settings() -> Settings:
    return Settings()


# --------------------------------------------------------------------------
# Anomaly thresholds (config.yaml)
#
# Kept out of the code so tuning a threshold is a config edit and a restart,
# not a code change. Defaults below mirror config.yaml so the system still runs
# if the file is missing.
# --------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

DEFAULT_ANOMALY_CONFIG: dict = {
    "resolution_time_outlier": {
        "enabled": True,
        "iqr_multiplier": 1.5,
        "min_group_size": 8,
        "severity_ratios": {"high": 2.0, "medium": 1.5},
    },
    "unresolved_high_priority": {
        "enabled": True,
        "priorities": ["Critical", "High"],
        "min_age_hours": 24,
        "severity_tiers": [
            {"hours": 168, "severity": "critical"},
            {"hours": 72, "severity": "high"},
            {"hours": 24, "severity": "medium"},
        ],
    },
    "response_sla_breach": {
        "enabled": True,
        "assumed_thresholds": True,
        "thresholds_hours": {"Critical": 1, "High": 2, "Medium": 4},
        "severities": {"Critical": "high", "High": "medium", "Medium": "low"},
    },
    "data_integrity_contradiction": {"enabled": True, "severity": "high"},
}

SEVERITY_ORDER = ["critical", "high", "medium", "low"]


@dataclass(frozen=True)
class AnomalyConfig:
    rules: dict
    severity_order: list[str]

    def rule(self, name: str) -> dict:
        return self.rules.get(name, {})

    def enabled(self, name: str) -> bool:
        return bool(self.rule(name).get("enabled", False))


def load_anomaly_config(path: str | Path | None = None) -> AnomalyConfig:
    """Read thresholds from config.yaml, falling back to the defaults above."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH

    if not path.exists():
        return AnomalyConfig(rules=deepcopy(DEFAULT_ANOMALY_CONFIG),
                             severity_order=list(SEVERITY_ORDER))

    try:
        import yaml

        loaded = yaml.safe_load(path.read_text()) or {}
    except Exception:  # unreadable or malformed: defaults keep the system up
        return AnomalyConfig(rules=deepcopy(DEFAULT_ANOMALY_CONFIG),
                             severity_order=list(SEVERITY_ORDER))

    rules = deepcopy(DEFAULT_ANOMALY_CONFIG)
    for name, overrides in (loaded.get("anomalies") or {}).items():
        if isinstance(overrides, dict):
            rules.setdefault(name, {}).update(overrides)

    return AnomalyConfig(
        rules=rules,
        severity_order=list(loaded.get("severity_order") or SEVERITY_ORDER),
    )
