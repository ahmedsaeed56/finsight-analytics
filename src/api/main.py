"""
src/api/main.py
===============

The HTTP surface over the graph.

    a request  ->  the same graph Streamlit calls  ->  JSON

WHAT THIS FILE IS NOT
---------------------
It is not a second implementation of anything. Every endpoint wraps a function
that already exists and is already tested — the graph, the pipeline, the
counters. If logic appears here that is not plumbing, it is in the wrong file.

The one place that rule bends is /explore, which computes aggregates the tool
layer has no function for. It is still pandas over the loaded frame and no
model is involved — see that section for why it exists at all.

NO DIRECT TOOL ENDPOINTS, DELIBERATELY
--------------------------------------
There is no POST /tools/aggregate_metric. Exposing the tools directly would
bypass the router, the confidence gate and the narrator — which is to say, all
of the architecture. A client that wants a specific tool asks a specific
question and the router picks it. GET /tools describes them so a client knows
what is askable; it does not let them be called.

SINGLE TENANT, DELIBERATELY
---------------------------
One dataset is loaded at a time, shared by every client. dataset.py holds it at
module level, and the alternative — a dataset id threaded through every tool
signature — is the design that file explicitly rejects: the router fills tool
parameters from an LLM, and an LLM has no dataset id to supply.

So a second upload replaces the first for everyone. That is a real limitation
and it is stated in /health rather than buried in a README.

SYNCHRONOUS ENDPOINTS, ON PURPOSE
---------------------------------
The graph is synchronous — LLM calls, pandas, sqlite. Declaring an endpoint
`async def` and then blocking inside it would freeze the event loop and serve
one request at a time. Plain `def` makes FastAPI run it in a threadpool
instead, which is the correct shape for blocking work.
"""

import sqlite3
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile

from src.api.models import (
    AskRequest,
    AskResponse,
    ColumnsResponse,
    DatasetInfo,
    ErrorResponse,
    ExploreRequest,
    ExploreResponse,
    GuardrailCounts,
    HealthResponse,
    PipelineReport,
    ThreadListResponse,
    ThreadResponse,
    ToolsResponse,
    VocabularyResponse,
)

# Beside the graph checkpoints Streamlit uses. Same file, so a thread started
# in one interface is visible from the other.
_CHECKPOINT_DB = "graph_checkpoints.db"

# Above this a numeric column is a measurement rather than a grouping. Twelve
# matches the rule the tool layer uses for listing values, so the API and the
# router agree about what counts as categorical.
_MAX_CATEGORICAL_LEVELS = 12

# Beyond this, an outlier list is a data dump rather than a chart annotation.
# One wild group should not be able to dominate the payload.
_MAX_OUTLIERS = 200

# Which tier a tool belongs to. Duplicated from build.py rather than imported
# because importing it would pull the whole graph module in at import time,
# and this file keeps project imports local — see lifespan.
_TIER = {
    "aggregate_metric": 1, "compare_groups": 1, "crosstab_rate": 1,
    "band_distribution": 1,
    "predict_default": 2, "predict_churn": 2, "score_population": 2,
    "simulate_loan": 2, "get_segment_profile": 2, "get_feature_importance": 2,
    "answer_freeform": 3,
}

_TOOL_DOCS = [
    ("aggregate_metric", "A rate or an average, overall or per group.",
     ["metric", "group_by", "aggfunc", "filters", "sort", "limit"]),
    ("compare_groups", "Whether a difference between groups is real, with a significance test. Outcomes only.",
     ["metric", "group_by", "groups", "filters"]),
    ("crosstab_rate", "A rate for every combination of two groupings — the confounder check.",
     ["metric", "row_by", "col_by", "filters"]),
    ("band_distribution", "How the book splits across the levels of one column.",
     ["group_by", "filters"]),
    ("predict_default", "Default risk for one named loan, with drivers.",
     ["loan_id"]),
    ("predict_churn", "Churn risk for one named customer, with drivers.",
     ["customer_id"]),
    ("score_population", "The whole book ranked by risk. Returns names, not a rate.",
     ["model", "limit"]),
    ("simulate_loan", "Score a loan that does not exist yet.",
     ["customer_id", "amount_pkr", "term_months", "purpose"]),
    ("get_segment_profile", "Which behavioural cluster a customer belongs to.",
     ["customer_id"]),
    ("get_feature_importance", "What a model relies on across the whole population.",
     ["model", "n"]),
    ("answer_freeform", "Questions no parameterised tool can express. Generates pandas at runtime.",
     ["question", "table"]),
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown.

    The graph is compiled HERE rather than per request — compiling per call
    would rebuild the whole thing on every question.

    PROJECT IMPORTS ARE LOCAL, not top-of-file. If build_graph raises on
    import — a missing parquet, an unset key — a module-level import kills the
    process before FastAPI can log anything. Inside lifespan, the traceback
    lands in the startup log where it can be read.
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    from src.graph.build import build_graph
    from src.memory.cache import clear_old

    # check_same_thread=False because sync endpoints run in a THREADPOOL —
    # different requests touch this connection from different threads, and
    # sqlite3 refuses that by default.
    connection = sqlite3.connect(_CHECKPOINT_DB, check_same_thread=False)

    app.state.graph = build_graph(checkpointer=SqliteSaver(connection))

    # Housekeeping, not invalidation — the cache key carries the fingerprint,
    # so nothing stale can be served. This only stops the table growing
    # forever. Startup is the natural place: sqlite has no scheduler.
    clear_old()

    yield

    connection.close()


app = FastAPI(
    title="JazzCash Analytics API",
    description=(
        "Natural-language analysis over a loan book. Every figure is computed "
        "in Python; the model routes questions to tools and narrates their "
        "results, and never calculates anything itself."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ==========================================================================
#  SYSTEM
# ==========================================================================

@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    """Is the server up, and can it actually answer anything?

    Two different questions, and a health check that only answers the first is
    close to useless. A process can be alive with no dataset loaded, in which
    case every question fails.
    """
    from src.tools.dataset import describe

    meta = describe()

    return {
        "status": "ok",
        "dataset_loaded": meta is not None,
        "dataset": meta,
        "single_tenant": True,
    }


# ==========================================================================
#  DATASET
# ==========================================================================

@app.get(
    "/dataset",
    response_model=DatasetInfo,
    tags=["dataset"],
    responses={404: {"model": ErrorResponse, "description": "Nothing loaded"}},
)
def get_dataset():
    """Which file the answers currently describe.

    404 rather than a null body: nothing is loaded, so the resource genuinely
    does not exist, and a client checking for 200 gets the right answer without
    inspecting the payload.
    """
    from src.tools.dataset import describe

    meta = describe()
    if meta is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "No dataset is loaded. POST /dataset/reference for the sample "
                "extract, or POST /dataset/upload with three CSVs."
            ),
        )
    return meta


@app.post("/dataset/reference", response_model=DatasetInfo, tags=["dataset"])
def load_reference_dataset():
    """Load the reference extract — the data the models were fitted on.

    Exists so the API is demoable with no upload. It is a FIXTURE, not a
    production path, and its label says so in every answer that mentions it.
    """
    from src.tools.dataset import describe, load_reference

    load_reference()
    return describe()


@app.post(
    "/dataset/upload",
    response_model=PipelineReport,
    tags=["dataset"],
    responses={400: {"model": ErrorResponse, "description": "Rejected"}},
)
def upload_dataset(
    customers: UploadFile = File(..., description="customers.csv"),
    loans: UploadFile = File(..., description="loans.csv"),
    transactions: UploadFile = File(..., description="transactions.csv"),
):
    """Three raw CSVs to three feature tables, with the report.

    MULTIPART, NOT JSON — files do not fit in a JSON body, so this endpoint has
    a different request shape from every other one here. Requires
    python-multipart to be installed.

    THE REPORT IS THE POINT. A pipeline that only says ok/failed hides what it
    did: which rows reconciled, which columns drifted, what anchor it derived.
    All of it comes back, because a user who cannot see the cleaning cannot
    trust the numbers that came out of it.

    A rejected upload returns 400 with the same report — the gate names the
    missing column, which is recoverable information rather than a dead end.
    """
    from src.pipeline.orchestrator import run_pipeline
    from src.tools.dataset import describe, load_dataset
    from src.tools.errors import ToolError

    # run_pipeline takes PATHS — a real upload lives in a buffer, so it lands
    # on disk first. A temp dir, because the frames are what matter and the
    # CSVs are not kept.
    tmp = Path(tempfile.mkdtemp())
    paths = []
    for upload, name in (
        (customers, "customers.csv"),
        (loans, "loans.csv"),
        (transactions, "transactions.csv"),
    ):
        path = tmp / name
        path.write_bytes(upload.file.read())
        paths.append(str(path))

    try:
        result = run_pipeline(*paths)
    except ToolError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        # A crash in cleaning or feature building. 400 rather than 500: the
        # cause is almost always the file, and the message names it.
        raise HTTPException(
            status_code=400,
            detail=f"Pipeline stopped: {type(exc).__name__} — {exc}",
        ) from exc

    if not result["ok"]:
        raise HTTPException(
            status_code=400,
            detail=f"Rejected at {result['failed_at']}. {result['report'].get('gate')}",
        )

    load_dataset(
        result["features"],
        label=result["label"],
        fingerprint=result["fingerprint"],
        as_of=result["as_of"],
    )

    return {
        "ok": True,
        "failed_at": None,
        "fingerprint": result["fingerprint"],
        "label": result["label"],
        "report": _jsonable(result["report"]),
        "dataset": describe(),
    }


# ==========================================================================
#  ASK
# ==========================================================================

@app.post(
    "/ask",
    response_model=AskResponse,
    tags=["ask"],
    responses={404: {"model": ErrorResponse, "description": "No dataset loaded"}},
)
def ask(payload: AskRequest, request: Request):
    """Put one question through the graph.

    The SAME graph the Streamlit interface calls — scope guardrail, follow-up
    rewriting, routing, the confidence gate, the tool, the retry edge, the
    narrator. Nothing is skipped for being an API call, which is the whole
    reason this endpoint is four lines of plumbing.

    The graph is reached through request.app.state rather than a global: it
    makes the dependency explicit and lets a test build an app with a stub
    graph in it.
    """
    from src.tools.dataset import describe

    meta = describe()
    if meta is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "No dataset is loaded, so there is nothing to answer about. "
                "POST /dataset/reference or /dataset/upload first."
            ),
        )

    state = {
        "question": payload.question,
        "thread_id": payload.thread_id,
        "fingerprint": meta.get("fingerprint"),
        "label": meta["label"],
        "retry_count": 0,
    }
    config = {"configurable": {"thread_id": payload.thread_id}}

    started = time.time()
    try:
        out = request.app.state.graph.invoke(state, config)
    except Exception as exc:
        # The graph catches its own expected failures — a bad parameter comes
        # back as a narrated error, not an exception. Reaching here means
        # something genuinely broke, so 500 is honest.
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc

    decision = out.get("decision") or {}
    execute_result = out.get("execute_result") or {}
    raw = execute_result.get("result")
    tool = decision.get("tool")

    return {
        "answer": out.get("answer") or "No answer was produced.",
        "tool": tool,
        "tier": _TIER.get(tool),
        "confidence": decision.get("confidence"),
        "result": _jsonable(raw),
        "expression": raw.get("expression") if isinstance(raw, dict) else None,
        "retries": out.get("retry_count", 0),
        "thread_id": payload.thread_id,
        "latency_s": round(time.time() - started, 2),
    }


# ==========================================================================
#  CONVERSATION
# ==========================================================================

@app.get("/threads", response_model=ThreadListResponse, tags=["ask"])
def list_threads(limit: int = 50):
    """Every conversation, most recent first.

    What lets a client show a chat list and reopen an old thread. The turns
    live in sqlite rather than in anyone's session, so a browser reload loses
    nothing — but only if something lists them, which is what this is.

    THE ONE ENDPOINT THAT TOUCHES SQL DIRECTLY. conversation.py has no
    list-threads function, and adding the query here rather than there is a
    deliberate short cut — worth moving down into that module, where the rest
    of the turn storage lives.

    `title` is the FIRST question in the thread. Nothing else in a stored
    conversation says what it was about without opening it.
    """
    from src.store import get_connection

    connection = get_connection()
    try:
        rows = connection.execute(
            """
            SELECT thread_id,
                   COUNT(*)      AS n_turns,
                   MIN(id)       AS first_id,
                   MAX(created_at) AS last_active
            FROM turn
            GROUP BY thread_id
            ORDER BY MAX(id) DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        # The title needs the first question per thread, which the grouped
        # query above cannot carry — GROUP BY gives one row per thread, and
        # the question belongs to a specific row inside it. So the ids come
        # first, then the questions.
        titles = {}
        if rows:
            ids = [row["first_id"] for row in rows]
            placeholders = ",".join("?" * len(ids))
            for row in connection.execute(
                f"SELECT id, question FROM turn WHERE id IN ({placeholders})",
                ids,
            ).fetchall():
                titles[row["id"]] = row["question"]
    except sqlite3.Error:
        # An unreadable history is not a reason to fail the request — the
        # client simply has no list to show.
        return {"threads": []}
    finally:
        connection.close()

    return {
        "threads": [
            {
                "thread_id": row["thread_id"],
                "title": titles.get(row["first_id"], "(untitled)"),
                "n_turns": row["n_turns"],
                "last_active": row["last_active"],
            }
            for row in rows
        ]
    }


@app.get("/threads/{thread_id}", response_model=ThreadResponse, tags=["ask"])
def get_thread(thread_id: str, n: int = 20):
    """The stored turns in one conversation, oldest first.

    Lets a client rebuild a chat after a reload — the turns live in sqlite, not
    in anyone's session.

    `thread_id` is a PATH parameter because it identifies which thread; `n` is
    a QUERY parameter because it filters how much of it. That split is the
    convention and it is worth following.

    DECLARED AFTER /threads, and that order matters: FastAPI matches routes in
    definition order, so a literal path must come before a parameterised one
    that could swallow it.
    """
    from src.memory.conversation import recent_turns

    return {"thread_id": thread_id, "turns": recent_turns(thread_id, n)}


# ==========================================================================
#  EXPLORE — direct charts, no model involved
# ==========================================================================

@app.get(
    "/columns",
    response_model=ColumnsResponse,
    tags=["explore"],
    responses={404: {"model": ErrorResponse, "description": "No dataset loaded"}},
)
def columns(table: str = "default"):
    """What can be charted, split into measurements and groupings.

    The client's selectboxes are populated from this. Sending the split rather
    than a flat list means the client cannot derive it differently from the
    server — which it would, eventually, and then a column would be groupable
    in one place and not the other.

    The categorical rule matches the tool layer's: object and category dtypes,
    plus any numeric column with twelve or fewer distinct values. A 0/1 flag is
    a grouping even though it is stored as an integer.
    """
    frame = _frame_or_404(table)

    numeric = frame.select_dtypes("number").columns.tolist()
    categorical = [
        column
        for column in frame.columns
        if frame[column].dtype == object
        or str(frame[column].dtype) == "category"
        or frame[column].nunique(dropna=True) <= _MAX_CATEGORICAL_LEVELS
    ]

    return {
        "table": table,
        "numeric": numeric,
        "categorical": categorical,
        "n_rows": int(len(frame)),
    }


@app.post(
    "/explore",
    response_model=ExploreResponse,
    tags=["explore"],
    responses={
        400: {"model": ErrorResponse, "description": "Bad column or chart type"},
        404: {"model": ErrorResponse, "description": "No dataset loaded"},
    },
)
def explore(payload: ExploreRequest):
    """Compute what one chart needs. Aggregates only, never rows.

    WHY THIS ENDPOINT EXISTS. The interface used to reach into the loaded
    dataframes and compute its own charts, which meant it could not run
    anywhere the data was not. Over HTTP that stops working — and shipping
    15,000 rows so a client can find a median would move a megabyte to produce
    one number.

    So the computation stays on this side and the WIRE CARRIES THE PLOT: bin
    edges and counts, five numbers per box, a correlation matrix. The one
    exception is the scatter, which is already a sample by design.

    NO MODEL IS INVOLVED. This is pandas over the loaded frame — unlike an
    /ask result there is no narration to check and nothing that could have been
    hallucinated. That is worth saying because the two look similar from a
    client's side and they are not the same kind of answer.
    """
    import numpy as np
    import pandas as pd

    frame = _frame_or_404(payload.table)
    kind = payload.chart_type

    base = {
        "chart_type": kind,
        "table": payload.table,
        "n_total": int(len(frame)),
    }

    # --- distribution --------------------------------------------------
    if kind == "distribution":
        column = _need(payload.column, "column", kind)
        series = _numeric_column(frame, column)

        counts, edges = np.histogram(series.dropna(), bins=payload.bins)
        return {
            **base,
            "bin_edges": [float(edge) for edge in edges],
            "counts": [int(count) for count in counts],
            "label": column,
        }

    # --- rate by group -------------------------------------------------
    if kind == "rate_by_group":
        metric = _need(payload.metric, "metric", kind)
        group_by = _need(payload.group_by, "group_by", kind)
        series = _numeric_column(frame, metric)
        _has_column(frame, group_by)

        grouped = series.groupby(frame[group_by], observed=True)
        aggregated = grouped.mean().sort_values(ascending=False)
        sizes = frame[group_by].value_counts()

        return {
            **base,
            "values": {str(k): _float(v) for k, v in aggregated.items()},
            # Sent alongside because a bar chart draws a 20% rate on 300 rows
            # exactly as tall as 20% on 3,000, and the client should be able
            # to say which is which.
            "group_sizes": {str(k): int(v) for k, v in sizes.items()},
            "label": f"mean {metric}",
        }

    # --- box by group --------------------------------------------------
    if kind == "box_by_group":
        metric = _need(payload.metric, "metric", kind)
        group_by = _need(payload.group_by, "group_by", kind)
        series = _numeric_column(frame, metric)
        _has_column(frame, group_by)

        boxes = []
        for group, values in series.groupby(frame[group_by], observed=True):
            values = values.dropna()
            if values.empty:
                continue

            q1, median, q3 = (float(v) for v in values.quantile([0.25, 0.5, 0.75]))
            iqr = q3 - q1

            # The standard 1.5 x IQR whiskers, and the same convention seaborn
            # uses — so a client drawing from these numbers produces the same
            # picture it used to produce from the raw rows.
            low_fence, high_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            inside = values[(values >= low_fence) & (values <= high_fence)]
            outside = values[(values < low_fence) | (values > high_fence)]

            boxes.append({
                "group": str(group),
                "min": float(inside.min()) if not inside.empty else q1,
                "q1": q1,
                "median": median,
                "q3": q3,
                "max": float(inside.max()) if not inside.empty else q3,
                # Capped: one heavy-tailed group should not be able to send
                # thousands of points and dominate the payload.
                "outliers": [float(v) for v in outside.head(_MAX_OUTLIERS)],
                "n": int(len(values)),
            })

        note = None
        trimmed = [b["group"] for b in boxes if len(b["outliers"]) >= _MAX_OUTLIERS]
        if trimmed:
            note = f"Outliers capped at {_MAX_OUTLIERS} for: {', '.join(trimmed)}."

        return {**base, "boxes": boxes, "label": metric, "note": note}

    # --- correlation ---------------------------------------------------
    if kind == "correlation":
        picked = payload.columns or []
        if len(picked) < 2:
            raise HTTPException(
                status_code=400,
                detail="A correlation matrix needs at least two columns.",
            )
        for column in picked:
            _numeric_column(frame, column)

        matrix = frame[picked].corr()

        return {
            **base,
            "matrix": {
                str(row): {str(col): _float(value) for col, value in values.items()}
                for row, values in matrix.to_dict("index").items()
            },
            # A dict does not guarantee order and a heatmap is unreadable if
            # the axes come back shuffled, so the order travels separately.
            "matrix_columns": [str(c) for c in picked],
            "label": "correlation",
        }

    # --- scatter -------------------------------------------------------
    if kind == "scatter":
        x = _need(payload.x, "x", kind)
        y = _need(payload.y, "y", kind)
        _numeric_column(frame, x)
        _numeric_column(frame, y)

        wanted = [x, y]
        if payload.hue:
            _has_column(frame, payload.hue)
            wanted.append(payload.hue)

        subset = frame[wanted].dropna()

        # random_state fixed so the same request draws the same picture. A
        # scatter that reshuffles on every rerun looks like the data changed.
        sampled = subset.sample(
            min(payload.sample, len(subset)), random_state=0
        ) if len(subset) > payload.sample else subset

        points = [
            {
                "x": float(row[x]),
                "y": float(row[y]),
                "hue": str(row[payload.hue]) if payload.hue else None,
            }
            for _, row in sampled.iterrows()
        ]

        return {
            **base,
            "points": points,
            "n_sampled": len(points),
            "label": f"{x} vs {y}",
            "note": (
                f"Sampled {len(points):,} of {len(subset):,} rows — a full "
                f"scatter at this size draws as a solid block, not a pattern."
            ) if len(points) < len(subset) else None,
        }

    # --- counts --------------------------------------------------------
    if kind == "counts":
        column = _need(payload.column, "column", kind)
        _has_column(frame, column)

        counts = frame[column].value_counts()
        shares = frame[column].value_counts(normalize=True)

        return {
            **base,
            "values": {str(k): float(v) for k, v in counts.items()},
            "shares": {str(k): _float(v) for k, v in shares.items()},
            "label": "rows",
        }

    # Unreachable — ChartType is a Literal, so FastAPI rejects anything else
    # with a 422 before this function runs. Here for completeness.
    raise HTTPException(status_code=400, detail=f"Unknown chart type '{kind}'.")


# ==========================================================================
#  OBSERVABILITY
# ==========================================================================

@app.get("/counters", response_model=GuardrailCounts, tags=["observability"])
def counters(days: int | None = None):
    """How often each of the five control points fired.

    The payoff for logging at all. "This system has five guardrails" is a
    claim; a count per guardrail is evidence — and a count stuck at zero is how
    you find one that never worked.

    `days` is optional because a cumulative total stops meaning much after a
    month: 47 blocks over what period?
    """
    from src.guardrails.counters import read_counts

    counts = read_counts(since_days=days)
    return {
        "window_days": days,
        "counts": counts,
        "total": sum(sum(actions.values()) for actions in counts.values()),
    }


# ==========================================================================
#  INTROSPECTION
# ==========================================================================

@app.get("/tools", response_model=ToolsResponse, tags=["introspection"])
def tools():
    """What this system can be asked, and what it cannot.

    Describes the tools; does not expose them. There is no endpoint that calls
    one directly — doing so would bypass the router, the confidence gate and
    the narrator, which is to say all of the architecture.
    """
    return {
        "tools": [
            {"tool": name, "tier": _TIER[name], "parameters": params,
             "description": doc}
            for name, doc, params in _TOOL_DOCS
        ],
        "note": (
            "Tools are not callable directly. Ask a question through /ask and "
            "the router selects one, with a confidence gate in front of it and "
            "a narrator behind it. Tier 1 measures what happened, Tier 2 is a "
            "model's estimate, Tier 3 generates pandas at runtime."
        ),
    }


@app.get(
    "/vocabulary",
    response_model=VocabularyResponse,
    tags=["introspection"],
    responses={404: {"model": ErrorResponse, "description": "No dataset loaded"}},
)
def get_vocabulary():
    """The metrics, groupings and allowed values in the LOADED file.

    Generated from the data, never stored — a file with a seventh region must
    be answerable about that region, and a hardcoded list of six never would
    be. This is the exact text the router is given.
    """
    from src.tools.analytics import vocabulary
    from src.tools.dataset import describe

    meta = describe()
    if meta is None:
        raise HTTPException(
            status_code=404,
            detail="No dataset is loaded, so there is no vocabulary to describe.",
        )

    return {"vocabulary": vocabulary(), "dataset": meta}


# ==========================================================================
#  HELPERS
# ==========================================================================

def _frame_or_404(table):
    """The requested feature table, or a readable failure.

    _frame raises ToolError when nothing is loaded — recoverable information
    written for a person, so it is passed through as a 404 rather than
    replaced with something generic.
    """
    from src.tools.errors import ToolError

    try:
        from src.tools.dataset import _frame
        return _frame(table)
    except ToolError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"'{table}' is not a table. Available: default, churn, segment.",
        ) from exc


def _need(value, name, kind):
    """A parameter this chart type cannot run without.

    ExploreRequest makes every chart field optional, because one model covers
    six shapes and a field required by one is meaningless to another. So the
    per-type requirement is checked here, with a message that names the chart
    rather than only the missing field.
    """
    if value is None:
        raise HTTPException(
            status_code=400,
            detail=f"'{kind}' needs a '{name}'.",
        )
    return value


def _has_column(frame, column):
    if column not in frame.columns:
        raise HTTPException(
            status_code=400,
            detail=f"'{column}' is not a column in this table.",
        )


def _numeric_column(frame, column):
    """A column that can be measured, or a readable refusal.

    Charting a text column as a number fails deep inside numpy with something
    a client cannot act on. Checked here instead, naming the column.
    """
    import pandas as pd

    _has_column(frame, column)
    series = frame[column]
    if not pd.api.types.is_numeric_dtype(series):
        raise HTTPException(
            status_code=400,
            detail=f"'{column}' is not numeric, so it cannot be measured on this chart.",
        )
    return series


def _float(value):
    """One float, JSON-safe.

    JSON has no NaN — json.dumps emits a bare NaN token that strict parsers
    reject. A correlation matrix produces them wherever a column has no
    variance, so this is not hypothetical.
    """
    import math

    number = float(value)
    return None if math.isnan(number) or math.isinf(number) else number


def _jsonable(value):
    """Make a tool return safe to serialise.

    pandas and numpy leak types JSON cannot encode — np.float64, np.int64,
    Timestamp, NaN. The tools convert most of them, but the pipeline report
    carries raw Timestamps and a defensive pass here is cheaper than a 500 on
    an upload that otherwise worked.

    Recursive because the shapes are nested: rates inside groups inside a
    crosstab.
    """
    import math

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value

    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]

    # numpy scalars carry .item(); Timestamps and anything else become strings.
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass

    return str(value) 