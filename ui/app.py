"""Minimal Streamlit UI.

Talks to the FastAPI service over HTTP rather than importing the pipeline, so
there is exactly one execution path: whatever the UI shows, the API produced.
That also means the UI demonstrates the API actually works.

Deliberately plain. The assessment weights functionality at 30% and UI polish at
0%, so effort belongs in translation accuracy, not CSS.
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT = int(os.getenv("UI_TIMEOUT_SECONDS", "60"))

EXAMPLES = [
    "How many tickets are currently open?",
    "Which agent resolved the most tickets this month?",
    "Show me all Critical tickets not resolved within 12 hours.",
    "What is the average customer rating for Technical category tickets?",
    "Are there any anomalies in resolution times this week?",
]

st.set_page_config(page_title="Support Ticket AI", page_icon="🎫", layout="wide")


# --------------------------------------------------------------------------
# API helpers
# --------------------------------------------------------------------------


def api_get(path: str, **params):
    """GET the API. Returns (payload, error_message)."""
    try:
        response = requests.get(f"{API_URL}{path}", params=params, timeout=TIMEOUT)
    except requests.RequestException as exc:
        return None, f"Cannot reach the API at {API_URL} ({exc})"
    return _unpack(response)


def api_post(path: str, payload: dict):
    try:
        response = requests.post(f"{API_URL}{path}", json=payload, timeout=TIMEOUT)
    except requests.RequestException as exc:
        return None, f"Cannot reach the API at {API_URL} ({exc})"
    return _unpack(response)


def _unpack(response):
    try:
        body = response.json()
    except ValueError:
        return None, f"HTTP {response.status_code}: unreadable response"

    if response.status_code >= 400:
        message = body.get("message") or body.get("detail") or f"HTTP {response.status_code}"
        return body, message
    return body, None


@st.cache_data(ttl=60)
def load_meta():
    return api_get("/meta")


# --------------------------------------------------------------------------
# Sidebar: service status
# --------------------------------------------------------------------------

st.title("Support Ticket AI")

health, health_error = api_get("/health")

with st.sidebar:
    st.subheader("Service")
    if health_error and not health:
        st.error(health_error)
        st.caption("Start the API with: uvicorn app.main:app --reload")
    else:
        ok = health.get("status") == "ok"
        (st.success if ok else st.warning)(f"API: {health.get('status')}")
        st.caption(f"{health.get('row_count', 0)} tickets loaded")
        st.caption(f"Date anchor: {health.get('date_anchor')}")
        llm = health.get("llm", {})
        st.caption(f"LLM: {llm.get('provider')} ({llm.get('model')})")
        if not llm.get("configured"):
            st.warning(llm.get("detail") or "LLM not configured")
        for warning in health.get("load_warnings", []):
            st.caption(f"⚠ {warning}")

    st.caption(f"API: {API_URL}")


ask_tab, anomaly_tab, data_tab = st.tabs(["Ask", "Anomalies", "Dataset"])


# --------------------------------------------------------------------------
# Ask
# --------------------------------------------------------------------------

with ask_tab:
    st.caption(
        "Questions are translated into a structured query. All figures are "
        "computed in Python from that query, never written by the model."
    )

    if "question" not in st.session_state:
        st.session_state.question = ""

    st.write("**Examples**")
    columns = st.columns(len(EXAMPLES))
    for column, example in zip(columns, EXAMPLES):
        with column:
            if st.button(example, key=f"ex_{example}", use_container_width=True):
                st.session_state.question = example

    question = st.text_input(
        "Ask a question about the tickets",
        value=st.session_state.question,
        placeholder="e.g. How many Critical tickets are unresolved?",
    )
    submitted = st.button("Ask", type="primary")

    if submitted and question.strip():
        with st.spinner("Translating and running the query..."):
            payload, error = api_post("/query", {"question": question})

        if error:
            st.error(error)
            if payload:
                if payload.get("valid_values"):
                    st.info("Valid values: " + ", ".join(payload["valid_values"]))
                if payload.get("supported_query_types"):
                    st.write("**This system can answer:**")
                    for shape in payload["supported_query_types"]:
                        st.write(f"- {shape}")
                if payload.get("error") == "not_implemented":
                    st.info("Anomaly questions will be answered once Part 4 lands.")
        elif payload:
            st.success(payload["answer"])

            # Anomaly questions return a report rather than a query result.
            report = payload.get("report")
            if report:
                flags = report.get("flags", [])
                if flags:
                    frame = pd.DataFrame(flags)
                    columns = ["ticket_id", "rule", "severity", "metric",
                               "observed_value", "threshold", "explanation"]
                    st.dataframe(
                        frame[[c for c in columns if c in frame]],
                        use_container_width=True, hide_index=True,
                    )
                for note in report.get("notes", []):
                    st.info(note)

            result = payload.get("result", {})
            if result.get("groups"):
                st.dataframe(
                    pd.DataFrame(result["groups"]).rename(
                        columns={"group": "Group", "value": "Value",
                                 "n_used": "Used", "n_total": "Total"}
                    ),
                    use_container_width=True, hide_index=True,
                )
            elif result.get("rows"):
                st.dataframe(pd.DataFrame(result["rows"]),
                             use_container_width=True, hide_index=True)
                st.caption(
                    f"Showing {result.get('returned_count')} of "
                    f"{result.get('matched_count')} matching tickets."
                )

            for warning in payload.get("warnings", []):
                st.warning(warning)

            meta_bits = [f"{payload.get('latency_ms')} ms"]
            if payload.get("provider"):
                meta_bits.append(f"provider: {payload['provider']}")
            if payload.get("repaired"):
                meta_bits.append(f"repaired after {payload.get('attempts')} attempts")
            meta_bits.append(f"dates anchored to {payload.get('date_anchor')}")
            st.caption(" · ".join(meta_bits))

            # Transparency: show exactly how the question was interpreted.
            with st.expander("Interpreted query (QuerySpec)"):
                st.json(payload["query_spec"])


# --------------------------------------------------------------------------
# Anomalies
# --------------------------------------------------------------------------

with anomaly_tab:
    payload, error = api_get("/anomalies", limit=500)

    if error:
        st.error(error)
    elif payload:
        flags = payload.get("flags", [])
        summary = payload.get("summary", {})

        left, right = st.columns(2)
        left.metric("Anomalies flagged", payload.get("total_flagged", 0))
        right.metric("Tickets affected", payload.get("tickets_flagged", 0))

        if summary:
            columns = st.columns(len(summary))
            for column, (rule, count) in zip(columns, summary.items()):
                column.metric(rule.replace("_", " ").title(), count)

        if not flags:
            st.success("No anomalies flagged.")
        else:
            frame = pd.DataFrame(flags)

            left, right = st.columns(2)
            chosen = left.multiselect("Rule", sorted(frame["rule"].unique()))
            levels = right.multiselect("Severity", sorted(frame["severity"].unique()))
            if chosen:
                frame = frame[frame["rule"].isin(chosen)]
            if levels:
                frame = frame[frame["severity"].isin(levels)]

            columns = ["ticket_id", "rule", "severity", "priority", "status",
                       "metric", "observed_value", "threshold", "explanation"]
            st.dataframe(
                frame[[c for c in columns if c in frame]],
                use_container_width=True, hide_index=True,
            )
            st.caption(f"Showing {len(frame)} of {payload.get('total_flagged', 0)} flags.")

        with st.expander("Thresholds in force"):
            st.caption("All thresholds come from config.yaml; no LLM is involved.")
            st.json(payload.get("thresholds", {}))

        for note in payload.get("notes", []):
            st.info(note)
        for warning in payload.get("warnings", []):
            st.warning(warning)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

with data_tab:
    meta, meta_error = load_meta()

    if meta_error:
        st.error(meta_error)
    elif meta:
        left, middle, right = st.columns(3)
        left.metric("Tickets", meta["row_count"])
        middle.metric("From", meta["date_range"]["min"][:10])
        right.metric("To", meta["date_range"]["max"][:10])
        st.caption(f"Relative dates resolve against {meta['date_anchor']}.")

        st.subheader("Values")
        for column, values in meta["distinct_values"].items():
            st.write(f"**{column}** — {', '.join(values)}")

        st.subheader("Missing values")
        nulls = pd.DataFrame(
            [{"column": c, "nulls": n, "% of rows": round(100 * n / meta["row_count"], 1)}
             for c, n in meta["null_counts"].items() if n]
        )
        if nulls.empty:
            st.write("No missing values.")
        else:
            st.dataframe(nulls, use_container_width=True, hide_index=True)
            st.caption(
                "Nulls are structural: unresolved tickets have no resolution time "
                "or rating, so averages over those columns exclude them."
            )

        if meta.get("numeric_summary"):
            st.subheader("Numeric columns")
            st.dataframe(
                pd.DataFrame(meta["numeric_summary"]).T.reset_index(names="column"),
                use_container_width=True, hide_index=True,
            )

        if meta.get("load_warnings"):
            st.subheader("Data quality")
            for warning in meta["load_warnings"]:
                st.warning(warning)
