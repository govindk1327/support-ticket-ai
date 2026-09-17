"""UI smoke test.

Boots the real API with uvicorn and the real Streamlit app as subprocesses, then
checks that the UI serves and that the endpoints it depends on answer over HTTP.

This is the test that would have caught "the UI imports the executor directly" or
"the API contract drifted from what the UI reads". It is marked `ui` so it can be
skipped in a fast run.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

pytestmark = pytest.mark.ui

ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait(url: str, timeout: float = 45.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=2).status_code < 500:
                return True
        except requests.RequestException:
            time.sleep(0.4)
    return False


def _terminate(process):
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


@pytest.fixture(scope="module")
def api_server():
    pytest.importorskip("uvicorn")
    port = _free_port()
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        if not _wait(f"{url}/health"):
            _terminate(process)
            pytest.fail("API did not start")
        yield url
    finally:
        _terminate(process)


@pytest.fixture(scope="module")
def ui_server(api_server):
    if shutil.which("streamlit") is None:
        pytest.skip("streamlit is not installed")

    port = _free_port()
    env = {**os.environ, "API_URL": api_server}
    process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "ui/app.py",
         "--server.port", str(port), "--server.headless", "true",
         "--browser.gatherUsageStats", "false"],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        if not _wait(f"{url}/_stcore/health", timeout=90):
            _terminate(process)
            output = process.stdout.read().decode()[-2000:] if process.stdout else ""
            pytest.fail(f"Streamlit did not start:\n{output}")
        yield url
    finally:
        _terminate(process)


# ------------------------------------------------------------------ API side


def test_api_starts_under_uvicorn(api_server):
    body = requests.get(f"{api_server}/health", timeout=10).json()
    assert body["dataset_loaded"] is True
    assert body["row_count"] == 500


def test_endpoints_the_ui_depends_on_answer(api_server):
    """Every endpoint ui/app.py calls must exist and respond."""
    assert requests.get(f"{api_server}/health", timeout=10).status_code == 200
    assert requests.get(f"{api_server}/meta", timeout=10).status_code == 200
    assert requests.get(f"{api_server}/anomalies", timeout=10).status_code == 200


def test_meta_contains_every_field_the_ui_reads(api_server):
    meta = requests.get(f"{api_server}/meta", timeout=10).json()
    for key in ("row_count", "date_range", "date_anchor", "distinct_values",
                "null_counts", "numeric_summary", "load_warnings"):
        assert key in meta, key
    assert "min" in meta["date_range"] and "max" in meta["date_range"]


def test_health_contains_every_field_the_ui_reads(api_server):
    health = requests.get(f"{api_server}/health", timeout=10).json()
    for key in ("status", "row_count", "date_anchor", "load_warnings", "llm"):
        assert key in health, key
    for key in ("provider", "model", "configured"):
        assert key in health["llm"], key


def test_query_without_a_key_fails_cleanly_not_with_a_crash(api_server):
    """No GROQ_API_KEY in this environment: the UI must get 503, not a 500."""
    response = requests.post(f"{api_server}/query",
                             json={"question": "how many tickets are open?"}, timeout=30)
    assert response.status_code in (200, 503)
    if response.status_code == 503:
        assert response.json()["error"] == "llm_unavailable"


# ------------------------------------------------------------------- UI side


def test_streamlit_app_starts(ui_server):
    assert requests.get(f"{ui_server}/_stcore/health", timeout=10).status_code == 200


def test_streamlit_serves_its_page(ui_server):
    response = requests.get(ui_server, timeout=15)
    assert response.status_code == 200
    assert "streamlit" in response.text.lower()


def test_anomalies_endpoint_returns_fields_the_ui_reads(api_server):
    body = requests.get(f"{api_server}/anomalies", params={"limit": 5}, timeout=15).json()
    for key in ("flags", "total_flagged", "tickets_flagged", "summary",
                "thresholds", "notes", "warnings"):
        assert key in body, key
    for key in ("ticket_id", "rule", "severity", "priority", "status",
                "metric", "observed_value", "threshold", "explanation"):
        assert key in body["flags"][0], key


def test_anomalies_work_without_an_api_key(api_server):
    """No LLM involved, so the Anomalies tab works even unconfigured."""
    response = requests.get(f"{api_server}/anomalies", params={"limit": 10}, timeout=15)
    assert response.status_code == 200
    assert response.json()["total_flagged"] == 286


def test_ui_talks_to_the_api_over_http_only():
    """The UI must not import the pipeline; one execution path, not two."""
    source = (ROOT / "ui" / "app.py").read_text()
    for forbidden in ("from app.", "import app.", "load_data_store", "execute("):
        assert forbidden not in source, f"UI should not reference {forbidden}"
    assert "requests.get" in source and "requests.post" in source
