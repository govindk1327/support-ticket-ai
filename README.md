# Support Ticket AI

Natural-language querying and anomaly detection over a 500-row customer support
ticket dataset.

---

## 1. Project overview

Ask questions about the support tickets in plain English and get answers computed
from the data, plus deterministic anomaly detection over the same dataset.

The system does four things, matching the four requirements in the brief:

| Requirement | How it is met |
|---|---|
| Ingest the CSV and make it queryable | Loaded once at startup into a pandas DataFrame with schema validation and derived metadata |
| Answer natural-language questions | An LLM translates the question into a validated `QuerySpec`; Python executes it |
| Detect and flag anomalies | Four deterministic rules with configurable thresholds; no LLM, no ML model |
| REST API **and** a minimal UI | FastAPI with four endpoints, plus a three-tab Streamlit UI that talks to it over HTTP |

The central design decision, and the one worth understanding first:

> **The LLM translates. Python computes.**

Every number a user sees is produced by pandas from a validated specification.
The model never sees the dataset, never writes code, and never produces a figure.

---

## 2. Architecture

```
┌──────────────────┐
│  Streamlit UI    │   three tabs: Ask · Anomalies · Dataset
└────────┬─────────┘
         │ HTTP (never imports the pipeline)
┌────────▼─────────┐
│   FastAPI app    │   /health  /meta  /query  /anomalies
└───┬──────────┬───┘
    │          └────────────────────────────┐
┌───▼─────────────────────┐      ┌──────────▼──────────┐
│   Query pipeline        │      │  Anomaly detector   │
│                         │      │   (pure Python)     │
│  translator ──► LLM     │      │                     │
│      │      (Groq/Ollama)│     │  4 rules, config.yaml
│      ▼                  │      │  no LLM involved    │
│  JSON → QuerySpec       │      └──────────┬──────────┘
│      ▼   (Pydantic)     │                 │
│  normalize_spec         │                 │
│      ▼                  │                 │
│  executor (pandas) ─────┼────────┐        │
└─────────────────────────┘        │        │
                          ┌────────▼────────▼────────┐
                          │  DataStore (DataFrame)   │
                          │  loaded once at startup  │
                          └──────────────────────────┘
```

**Request flow for `POST /query`:**

1. Question is length-checked and stripped — before any LLM call.
2. The translator builds a prompt from live dataset metadata (schema, real
   categorical values, null counts, date anchor) and calls the model at
   `temperature=0` with JSON output enforced.
3. The response is parsed tolerantly (markdown fences, leading prose), then
   validated by Pydantic, then normalized against the dataset's real values.
4. Any failure triggers **one** repair retry with the specific error fed back.
5. The validated spec is executed by pandas. **No LLM involvement from here on.**
6. The answer sentence is assembled in Python from the computed result.

Anomaly questions branch at step 5 to the detector instead of the executor. The
translator still interprets the question (filters and time window); the rules do
the detection. There is no second LLM call anywhere in the system.

---

## 3. Why this architecture

The dataset is 500 rows, ten columns, with 26 distinct `issue_summary` strings.
That fact drives every choice below.

**No vector database or RAG.** RAG solves the problem of finding relevant
context in a corpus too large for a prompt. Here the entire schema and every
categorical value fit in about 1,200 tokens, and the questions are aggregations
over structured columns, not retrieval over text. Embedding 26 templated strings
would add a dependency, an index and a failure mode to solve a problem that does
not exist.

**No SQL database.** 500 rows fit comfortably in memory, and pandas gives us
groupby and quartiles directly. Adding SQL would mean an LLM writing query
strings — an injection surface and a class of errors that only appear at
execution time — to replace something already correct.

**No LangChain or LangGraph.** The integration is one HTTP call, one prompt and
one Pydantic model: roughly sixty lines. A framework would add indirection and
version churn without removing any of that work.

**No agents.** Translation is a single step with no tool loop. An agent
architecture would introduce nondeterminism and latency to a problem that is
deterministic once the spec exists.

**A constrained intermediate representation instead of generated code.** The
model emits a small, fixed JSON object describing *what to compute*; Python
decides *how*. This bounds what the system can be asked to do, makes every
request inspectable, and gives a clean error path when a question falls outside
the envelope. It is also why injection attempts fail at validation: there is no
field in the spec that can carry code.

These are deliberate choices for this scale, not omissions. Section 12 covers
what would change at a larger one.

---

## 4. LLM usage

### What the LLM does

Exactly one thing: turn a natural-language question into a `QuerySpec`.

```
"How many Critical tickets are unresolved?"
        ↓
{"intent": "aggregate", "metric": "count",
 "filters": [{"field": "priority", "op": "eq", "value": "Critical"},
             {"field": "status",   "op": "ne", "value": "Resolved"}]}
```

### What the LLM does not do

- It never sees ticket data — only the schema and the distinct values.
- It never writes Python, SQL or pandas expressions.
- It never computes dates. It emits a preset name such as `this_month`; all date
  arithmetic happens in `app/query/dates.py`.
- It never produces a number or writes the final answer sentence.
- It is not involved in anomaly detection at all.

### The QuerySpec

| Field | Meaning |
|---|---|
| `intent` | `aggregate` · `list` · `anomaly` · `unsupported` |
| `metric` | `count` · `avg` · `sum` · `min` · `max` · `median` |
| `target` | numeric column (required for every metric except `count`) |
| `filters` | AND-combined conditions |
| `or_groups` | each group is OR-internal, ANDed with the rest |
| `group_by` | a categorical column |
| `date_range` | a preset name or an explicit start/end |
| `sort`, `limit` | ordering and truncation |
| `reason` | required when `intent` is `unsupported` |

`or_groups` exists for one specific reason. *"Critical tickets not resolved
within 12 hours"* means `priority = Critical AND (status ≠ Resolved OR
resolution_time_hrs > 12)`. Reading it as a plain AND returns **3** tickets;
the correct reading returns **34**. The 31 it would miss are the still-open
tickets — exactly the ones worth seeing.

### Validation, in four layers

1. **Prompt** — schema, real categorical values, null counts and the date anchor,
   built from the loaded DataStore so there is one source of truth.
2. **Parse** — tolerant of markdown fences and leading prose, because
   instruction-tuned models produce both. Tolerance stops there.
3. **Pydantic** — rejects unknown columns, invented operators, type mismatches,
   incoherent metric/target pairs and smuggled extra fields.
4. **Normalization** — resolves values against the dataset: exact, then
   case-insensitive, then unique prefix. `"critical"` → `Critical`,
   `"tech"` → `Technical`. There is deliberately **no** fuzzy matching: a guess
   like `"Hugh"` → `"High"` would produce a confident wrong number. An
   unresolvable value is rejected with the valid options listed, so
   *"how many Urgent tickets?"* never answers `0 tickets`.

### Repair

One retry, with the specific validation error handed back to the model. Capped
at one on purpose: a model that cannot fix a named error in one turn is unlikely
to fix it in three, and each turn costs a full round trip. This is not an agent
loop — the maximum is two calls, ever.

### Provider fallback

Groq primary, Ollama fallback. **Only** an unreachable provider triggers the
fallback — network errors, timeouts, HTTP errors, a missing key. A model that
returns nonsense does *not*: switching providers cannot fix bad output, and the
repair retry already handles it.

### Hallucination prevention

The model produces a description of a computation, not an answer. If the
description is invalid it is rejected before anything runs. If it is valid but
misreads the question, the interpreted spec is returned in every response and
shown in the UI, so the user can see exactly how their question was read.

---

## 5. Anomaly detection

Four deterministic rules. No LLM, no Isolation Forest, no learned model. Every
flag carries the observed value and the threshold it crossed, so a support lead
can check the arithmetic — which is precisely why this is rules rather than an
unsupervised score.

All thresholds live in [`config.yaml`](config.yaml). Rule names are stable
identifiers used by the API filter, the UI and the tests.

### Rule 1 — `resolution_time_outlier` (18 flags)

`resolution_time_hrs > Q3 + 1.5 × IQR`, computed **per priority** on clean
resolved rows, minimum group size 8.

| Priority | Q3 | IQR | Threshold | n |
|---|---|---|---|---|
| Critical | 5.80 | 2.08 | **8.91h** | 14 |
| High | 10.40 | 5.60 | **18.80h** | 72 |
| Medium | 20.35 | 11.43 | **37.49h** | 112 |
| Low | 38.50 | 24.00 | **74.50h** | 101 |

Per-priority matters: the Critical threshold is 8.9h and the Low threshold is
74.5h. A single global cutoff would over-flag Low and under-flag Critical.

IQR rather than mean + 3σ because the distribution is heavily right-skewed
(mean 19.2, median 12.0); a 3σ cutoff lands near 79h and catches almost nothing.

Rows flagged by Rule 4 are excluded from this calculation — including them
shifts the Critical threshold from 8.913 to 9.775.

### Rule 2 — `unresolved_high_priority` (80 flags)

Critical/High, not Resolved, older than **24h** (configurable), measured from the
dataset anchor. Severity tiers at 24h / 72h / 168h. Sorted oldest first.

> **Documented limitation.** This dataset is a static Q1-2024 snapshot, so every
> unresolved High/Critical ticket is months old and **all 80 trip this rule**.
> On live data it would be a genuine alert; here it is effectively a backlog
> listing. It is tiered and sorted so the output remains usable, and the API
> returns this caveat in its `notes` field rather than hiding it.

### Rule 3 — `response_sla_breach` (160 flags)

First response slower than the target for that priority.

| Priority | Target | Breaches |
|---|---|---|
| Critical | > 1h | 46 of 55 |
| High | > 2h | 77 of 134 |
| Medium | > 4h | 37 of 169 |
| Low | no target | — |

> **These thresholds are assumptions.** They are reasonable support-desk
> defaults invented for this assessment, **not real customer SLA data**. They are
> labelled as assumed in `config.yaml`, in the API `notes`, and in the text of
> every individual flag.
>
> They also fit this dataset poorly. `response_time_hrs` is uniformly distributed
> between 0.2 and 5.0 with no relationship to priority, so a 1-hour Critical
> target catches 84% of Critical tickets. The honest reading is that the
> synthetic data was not generated with any SLA in mind. The thresholds were left
> at defensible values rather than tuned backwards to produce a prettier number.

This is the only rule with full coverage: `response_time_hrs` has no nulls, so it
applies to all 500 tickets rather than only resolved ones.

### Rule 4 — `data_integrity_contradiction` (28 flags)

`resolution_time_hrs < response_time_hrs` — resolved before the first response.
Impossible by definition, so zero false positives. Flagged rather than dropped,
because silently discarding rows hides a real data problem.

### Totals and filtering

**286 flags across 206 of 500 tickets** (71 critical, 92 high, 83 medium, 40 low).
That volume is a direct consequence of Rules 2 and 3 above.

Anomaly detection accepts the same filters as a normal query, reusing the
executor's filter code so `eq`, `contains` and `or_groups` mean the same thing in
both places. Rule 1's IQR threshold is then computed from the **filtered**
population — outliers among Critical tickets are judged against Critical tickets.

Filtering on `resolution_time_hrs` or `response_time_hrs` is **rejected with
HTTP 422**, not silently ignored. Those are the columns Rules 1 and 3 measure:
narrowing to `resolution_time_hrs > 50` and then asking which resolution times
are outliers computes the threshold from an already-truncated population. That is
circular, and returning a confident number would be worse than refusing.

> **Note on `customer_rating`.** It is permitted as an anomaly filter, since no
> rule derives a threshold from it. But it is null for every unresolved ticket in
> this dataset, so filtering by it **excludes all of Rule 2's candidates**. The
> result is correct but narrower than it may appear.

If filtering shrinks a priority group below the minimum size, Rule 1 skips that
group and the response says so — otherwise an empty result would look like a
clean bill of health.

---

## 6. API

Interactive docs at `http://localhost:8000/docs`.

### `GET /health`

Dataset state, row count, date range and anchor, load warnings, and LLM provider
configuration. Returns **200 even when degraded** — the point of a health check
is to say what is wrong, not to refuse to answer.

```bash
curl http://localhost:8000/health
```
```json
{
  "status": "ok",
  "dataset_loaded": true,
  "row_count": 500,
  "date_anchor": "2024-03-30",
  "load_warnings": ["28 ticket(s) have resolution_time_hrs < response_time_hrs ..."],
  "llm": {"provider": "groq", "model": "openai/gpt-oss-120b",
          "configured": true, "reachable": null}
}
```

`?probe=true` makes a live provider call and sets `reachable`. It is **opt-in**:
health checks get polled, and a live call costs latency and free-tier quota, so
normal checks do not depend on the provider being up.

### `GET /meta`

Columns, distinct values, null counts, date range, numeric summary. This is the
same metadata the translation prompt is built from, so the UI and the model never
disagree about what the dataset contains.

### `POST /query`

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the average customer rating for Technical tickets?"}'
```
```json
{
  "answer": "The average customer rating where category is Technical is 3.74.",
  "result": {"value": 3.74, "n_used": 104, "n_total": 152},
  "query_spec": {"intent": "aggregate", "metric": "avg",
                 "target": "customer_rating",
                 "filters": [{"field": "category", "op": "eq", "value": "Technical"}]},
  "date_anchor": "2024-03-30",
  "warnings": ["Computed over 104 of 152 matching tickets; 48 have no customer rating (unresolved)."],
  "repaired": false, "attempts": 1, "latency_ms": 480
}
```

Aggregates over nullable columns always report `n_used` / `n_total` and warn.
35% of tickets have no resolution time or rating, so an unqualified average would
quietly exclude a third of the data.

### `GET /anomalies`

```bash
curl "http://localhost:8000/anomalies?rule=data_integrity_contradiction&limit=5"
curl "http://localhost:8000/anomalies?severity=critical&limit=10"
```

`rule` and `severity` accept comma-separated values. `limit` is 1–500 and
truncates the flag list without distorting the counts. Works with no API key —
detection is entirely deterministic.

### Status codes

| Code | Meaning |
|---|---|
| 200 | Success (including a degraded `/health`) |
| 400 | Bad question — empty or over the length limit |
| 422 | Translation failed after the repair retry; unsupported question; or an anomaly filter that would distort a rule |
| 503 | No LLM provider reachable — `/health`, `/meta` and `/anomalies` still work |
| 500 | Unexpected error, with a request id and no traceback |

---

## 7. UI

```bash
streamlit run ui/app.py        # needs the API running
```

Opens at `http://localhost:8501`. Three tabs:

- **Ask** — example buttons for the brief's five sample queries, a question box,
  the answer, a results table, warnings, and an expander showing the interpreted
  `QuerySpec`.
- **Anomalies** — total flagged, per-rule counts, rule and severity filters, the
  flag table with observed values and thresholds, and the thresholds in force.
- **Dataset** — row count, date range, distinct values, null summary, numeric
  summary, data-quality warnings.

The UI calls the API over HTTP and never imports the pipeline, so there is
exactly one execution path and the UI demonstrates that the API works. A test
enforces this by inspecting the source.

It is deliberately plain: the brief weights functionality at 30% and UI polish at
0%, so the effort went into translation accuracy.

---

## 8. Setup

> **Python 3.10 or newer is required.** The codebase uses `X | None` type
> annotations (PEP 604). On 3.9 the app fails immediately with an explanatory
> message. Check yours with `python --version` (Windows) or `python3 --version`.

### macOS / Linux

```bash
# 1. Install
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure (see section 9 — or skip and use Ollama)
cp .env.example .env
#    then edit .env and paste your free Groq key

# 3. Run both services
./start.sh
```

### Windows (PowerShell)

```powershell
# 1. Install
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Configure
copy .env.example .env
notepad .env          # paste your free Groq key

# 3. Run both services (opens two windows)
.\start.bat
```

If PowerShell blocks the activation script, allow it for the current session
only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Both launchers resolve their own directory, so they work from any working
directory. API → `http://localhost:8000` (docs at `/docs`), UI →
`http://localhost:8501`.

### Running the services separately

```bash
uvicorn app.main:app --reload          # terminal 1
streamlit run ui/app.py                # terminal 2
```

On Windows, set `API_URL` in the UI terminal first if you changed the API port:
`$env:API_URL="http://127.0.0.1:8000"`.

Developed on Python 3.12. The suite passes on both pandas 2.2 / numpy 2.2 and
pandas 3.0 / numpy 2.4.

If `python --version` reports 3.9 or older, recreate the environment with a
newer interpreter — on Windows the `py` launcher makes this easy:

```powershell
deactivate                      # if the old venv is active
Remove-Item -Recurse -Force .venv
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 9. Configuration

### Zero cost, two options

**Groq (default, recommended).** Free tier, **no credit card**, gated only by
rate limits. Get a key at <https://console.groq.com/keys> and put it in `.env`.

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

> **Model note.** `openai/gpt-oss-120b` is a Groq *production* model reachable on
> the free tier, verified against the Groq docs in September 2026.
> `openai/gpt-oss-20b` is faster and lighter if rate limits bite.
> `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` are **Enterprise-only**
> and will fail without a sales contract — do not use them.

**Ollama (fully local, no account at all).**

```bash
# install from https://ollama.com
ollama pull llama3.1:8b
```
```env
LLM_PROVIDER=ollama
LLM_FALLBACK_PROVIDER=none
```

By default Groq is primary and Ollama is the fallback, used only when Groq is
unreachable.

### With no key at all

The API still starts. `/health`, `/meta` and `/anomalies` work fully — anomaly
detection never touches a model. Only `/query` needs a provider, and it returns
a clear **503** rather than crashing. `/health` reports `degraded` and says why.

### Environment variables

All optional except the key. Full list with comments in `.env.example`.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `groq` | Primary provider |
| `LLM_FALLBACK_PROVIDER` | `ollama` | Used only if the primary is unreachable; `none` disables |
| `GROQ_API_KEY` | — | Free key from the Groq console |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Free-tier production model |
| `OLLAMA_MODEL` | `llama3.1:8b` | Local model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama endpoint |
| `LLM_TIMEOUT_SECONDS` | `30` | Per-request timeout |
| `MAX_QUESTION_LENGTH` | `500` | Guards against prompt stuffing |
| `API_URL` | `http://localhost:8000` | Where the UI looks for the API |

### `config.yaml`

All anomaly thresholds, kept in version control so tuning is a config edit and a
restart rather than a code change. `.env` is git-ignored; `.env.example` contains
placeholders only.

---

## 10. Testing

```bash
pytest                    # everything
pytest -m "not ui"        # skip subprocess tests (faster)
```

**387 tests, all passing.** No API key and no network required — the LLM is
replaced by scripted stubs, so what is tested is the pipeline's handling of model
output (good, malformed, hallucinated and hostile) rather than the model itself.

| File | Tests | Covers |
|---|---|---|
| `test_anomalies.py` | 82 | four rules, exact thresholds, boundaries, IQR exclusion, filtering, determinism |
| `test_translator.py` | 69 | sample queries, wording variants, malformed JSON, hallucinated columns, injection, repair |
| `test_api.py` | 50 | endpoints, status codes, error shapes, security |
| `test_executor.py` | 45 | operators, aggregation, grouping, OR-groups, null denominators |
| `test_normalization.py` | 36 | exact / case-insensitive / prefix / unknown / ambiguous |
| `test_validation.py` | 33 | schema rejection of bad specs |
| `test_llm_client.py` | 27 | both providers, fallback behaviour |
| `test_dates.py` | 21 | every preset, month/quarter/leap boundaries |
| `test_data_store.py` | 14 | load, schema failures, integrity warnings |
| `test_ui_smoke.py` | 10 | boots real uvicorn + Streamlit, contract checks |

Expected values are hand-computed from the CSV rather than recomputed with the
implementation's own logic — a test that builds its expectation by calling the
code under test would pass even if the formula were wrong.

---

## 11. Limitations


**Dataset is a historical snapshot.** Tickets run 2024-01-01 to 2024-03-30.
Relative dates resolve against **2024-03-30**, not today. Anchoring to wall-clock
time would make every "this week" query return zero rows. Every response includes
`date_anchor` so this is visible rather than surprising.

**Rule 2 fires on all 80 unresolved High/Critical tickets** because of that
snapshot. See section 5.

**Rule 3's SLA thresholds are assumptions**, and they fit this synthetic data
poorly. See section 5.

**Rule 1 cannot evaluate unresolved tickets.** Their `resolution_time_hrs` is
null, so a ticket open for months is invisible to the outlier rule. Rule 2
partially covers the gap. This is structural, not a bug.

**Anomaly volume is a function of the configured rules.** 286 flags reflects
these four rules at these thresholds, not an objective count of "things wrong".

**The dataset is synthetic and templated.** 26 distinct `issue_summary` strings
across 500 rows, and `response_time_hrs` is uniform with no relation to priority.
Free-text search is substring matching only; semantic search would add nothing
here.

**Query expressiveness is bounded by the QuerySpec.** No joins, no subqueries, no
arbitrary derived columns. A genuinely novel question shape needs a schema
extension. Out-of-scope questions return `intent=unsupported` with a reason
rather than a forced wrong answer.

**Translation accuracy is model-dependent.** The pipeline is tested exhaustively;
whether a given model produces good specs is a separate question. `temperature=0`
reduces variance but does not guarantee identical specs across runs — the
executor is fully deterministic given a spec, the translation step is not.

**Single-process, in-memory.** No persistence, no auth, no rate limiting, no
horizontal scaling. Appropriate for a 500-row assessment; **this is not
production-ready** and is not claimed to be.

---

## 12. What would change at scale


- **500 rows → 10M:** pandas moves to DuckDB or Postgres; the `QuerySpec` becomes
  a parameterised SQL builder. The IR and the validation boundary survive
  unchanged — that is the main benefit of not having the LLM write SQL directly.
- **CSV → live database:** the DataStore becomes a connection with a query cache;
  rules run on a schedule rather than per request; the date anchor becomes
  `now()`, which fixes Rule 2's snapshot problem.
- **Higher accuracy requirements:** build a labelled golden set of
  question → spec pairs, measure translation accuracy, and use it as a regression
  gate when swapping models.
- **Anomaly detection with history:** keep the rules as the explainable floor and
  add a learned layer above once there is enough history and analyst feedback to
  validate it against.

---

## Project structure

```
support-ticket-ai/
├── app/
│   ├── main.py              FastAPI app, routes, error handlers
│   ├── config.py            settings + config.yaml loading
│   ├── models.py            QuerySpec, AnomalyFlag, errors
│   ├── data_store.py        CSV load, metadata, value normalization
│   ├── llm/
│   │   ├── client.py        Groq + Ollama + fallback
│   │   └── prompts.py       prompt built from live metadata
│   ├── query/
│   │   ├── translator.py    question → validated QuerySpec
│   │   ├── executor.py      QuerySpec → result (pure pandas)
│   │   └── dates.py         preset resolution against the anchor
│   └── anomalies/
│       ├── rules.py         the four rules
│       └── detector.py      orchestration, filtering, ordering
├── ui/app.py                Streamlit UI (HTTP only)
├── tests/                   387 tests
├── data/support_tickets.csv
├── config.yaml              anomaly thresholds
├── requirements.txt
├── .env.example
├── start.sh                 macOS / Linux launcher
└── start.bat                Windows launcher
```
