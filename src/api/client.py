"""
src/api/client.py
=================

How a client reaches the system — over HTTP, or in-process.

    the same nine calls, two transports

WHY THIS FILE EXISTS
--------------------
Without it, `requests.post(...)` appears in the middle of the interface code,
once per feature, and the URL is written out eight times. Worse, the choice
between calling the API and calling the graph directly would be made eight
times too — and eventually differently in two of them.

So the transport lives here and nowhere else. The interface calls
`client.ask(...)` and does not know or care which side of a socket the answer
came from.

TWO TRANSPORTS, ONE INTERFACE
-----------------------------
HTTP mode is the real architecture: the interface is a client, the graph runs
once behind an API, and any number of clients can share it. That is what
deployment looks like and what the API exists for.

Local mode imports the graph and calls it in the same process. It exists
because a demo that requires two terminals is a demo that sometimes does not
run, and because the interface should work with nothing else started.

Set JAZZCASH_USE_API=true for HTTP; anything else is local. The functions
below behave identically either way — same arguments, same returned shapes —
which is the whole point of putting them in one file.

WHAT LOCAL MODE COSTS
---------------------
It keeps the import coupling HTTP mode removes: local mode reaches into
src.graph, src.pipeline and src.tools, so the interface is not really a client
in that mode. It is a deliberate fallback, not the target architecture, and
the README should say so.

ERRORS
------
Both transports raise ApiError, so the interface writes one except clause.
An HTTP failure carries the server's `detail` — which for a ToolError is a
message written for a person, naming the bad value and the valid ones, and
worth passing through intact rather than replacing with "request failed".
"""

import os
from typing import Any, Optional

DEFAULT_BASE_URL = "http://127.0.0.1:8000"

# Generous, because a Tier 3 question can take three LLM calls and an upload
# runs the whole cleaning pipeline. A short timeout here would look like a
# server failure when it was only a slow answer.
ASK_TIMEOUT = 180
UPLOAD_TIMEOUT = 300
QUICK_TIMEOUT = 30


class ApiError(Exception):
    """Anything that stopped a call from returning an answer.

    `detail` is the server's message where there is one. `status` is the HTTP
    code, or None in local mode — which is how a caller can tell a 404 (no
    dataset) from a 500 (something broke) without parsing text.
    """

    def __init__(self, detail, status=None):
        super().__init__(detail)
        self.detail = detail
        self.status = status


# ==========================================================================
#  MODE
# ==========================================================================

def use_api():
    """Is this client talking HTTP, or calling the graph directly?

    Read on every call rather than cached at import, so a running Streamlit
    session can flip the toggle without a restart.
    """
    return os.environ.get("JAZZCASH_USE_API", "").strip().lower() == "true"


def base_url():
    """Where the API lives.

    An environment variable rather than a constant so pointing at a deployed
    instance is configuration, not a code change.
    """
    return os.environ.get("JAZZCASH_API_URL", DEFAULT_BASE_URL).rstrip("/")


def mode_label():
    """One line for an interface to display.

    Worth showing: the two modes answer identically but fail completely
    differently, and a user staring at a stalled app should be able to see
    which one they are in.
    """
    return f"API — {base_url()}" if use_api() else "in-process"


# ==========================================================================
#  HTTP PLUMBING
# ==========================================================================

def _request(method, path, timeout=QUICK_TIMEOUT, **kwargs):
    """One HTTP call, with the server's error message preserved.

    FastAPI puts a readable message in `detail` on every 4xx this API raises —
    the missing column, the six real regions, the reason a file was rejected.
    Discarding that in favour of a status code would throw away the most
    useful thing in the response.
    """
    import requests

    url = f"{base_url()}{path}"

    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
    except requests.exceptions.ConnectionError as exc:
        raise ApiError(
            f"Cannot reach the API at {base_url()}. Is it running? "
            f"Start it with: python -m uvicorn src.api.main:app"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise ApiError(
            f"The API did not respond within {timeout}s. A Tier 3 question can "
            f"be slow, but this is beyond that."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise ApiError(f"Request failed: {exc}") from exc

    if response.status_code >= 400:
        detail = f"HTTP {response.status_code}"
        try:
            body = response.json()
            if isinstance(body, dict) and "detail" in body:
                # 422 sends a LIST of validation errors rather than a string —
                # a shape mismatch, which is a client bug worth seeing whole.
                detail = body["detail"] if isinstance(body["detail"], str) else str(body["detail"])
        except ValueError:
            detail = response.text or detail
        raise ApiError(detail, status=response.status_code)

    return response.json()


# ==========================================================================
#  SYSTEM
# ==========================================================================

def health():
    """Is the backend up, and does it have data?

    In local mode there is no server to be down, so this reports on the
    process itself — which keeps the interface's status display meaningful in
    both modes rather than blank in one.
    """
    if use_api():
        return _request("GET", "/health")

    from src.tools.dataset import describe

    meta = describe()
    return {
        "status": "ok",
        "dataset_loaded": meta is not None,
        "dataset": meta,
        "single_tenant": True,
    }


def is_reachable():
    """A cheap yes/no for a status indicator.

    Never raises — a status light that throws is worse than no status light.
    """
    try:
        health()
        return True
    except ApiError:
        return False


# ==========================================================================
#  DATASET
# ==========================================================================

def get_dataset():
    """What is loaded, or None.

    The API answers 404 when nothing is loaded, which is correct HTTP and
    unhelpful here — the interface asks this on every rerun and "nothing yet"
    is an ordinary state, not an error. So the 404 becomes None and the two
    modes return the same thing.
    """
    if use_api():
        try:
            return _request("GET", "/dataset")
        except ApiError as exc:
            if exc.status == 404:
                return None
            raise

    from src.tools.dataset import describe

    return describe()


def load_reference():
    """Point the system at the reference extract."""
    if use_api():
        return _request("POST", "/dataset/reference", timeout=QUICK_TIMEOUT)

    from src.tools.dataset import describe, load_reference as _load_reference

    _load_reference()
    return describe()


def upload_dataset(customers_bytes, loans_bytes, transactions_bytes):
    """Three CSVs through the full pipeline.

    Takes BYTES rather than paths or file objects, because the two transports
    need different things — HTTP needs a multipart body, the orchestrator needs
    files on disk — and bytes is what both can be built from. The interface
    hands over `uploaded_file.getvalue()` and stays out of it.

    Returns the pipeline report either way: gate, reconciliation, drift.
    """
    if use_api():
        files = {
            "customers": ("customers.csv", customers_bytes, "text/csv"),
            "loans": ("loans.csv", loans_bytes, "text/csv"),
            "transactions": ("transactions.csv", transactions_bytes, "text/csv"),
        }
        return _request("POST", "/dataset/upload", timeout=UPLOAD_TIMEOUT, files=files)

    import tempfile
    from pathlib import Path

    from src.pipeline.orchestrator import run_pipeline
    from src.tools.dataset import describe, load_dataset
    from src.tools.errors import ToolError

    tmp = Path(tempfile.mkdtemp())
    paths = []
    for content, name in (
        (customers_bytes, "customers.csv"),
        (loans_bytes, "loans.csv"),
        (transactions_bytes, "transactions.csv"),
    ):
        path = tmp / name
        path.write_bytes(content)
        paths.append(str(path))

    try:
        result = run_pipeline(*paths)
    except ToolError as exc:
        raise ApiError(str(exc)) from exc
    except Exception as exc:
        raise ApiError(f"Pipeline stopped: {type(exc).__name__} — {exc}") from exc

    if not result["ok"]:
        raise ApiError(f"Rejected at {result['failed_at']}.")

    load_dataset(
        result["features"],
        label=result["label"],
        fingerprint=result["fingerprint"],
        as_of=result["as_of"],
        paths=result["paths"],
    )

    return {
        "ok": True,
        "failed_at": None,
        "fingerprint": result["fingerprint"],
        "label": result["label"],
        "report": result["report"],
        "dataset": describe(),
    }


# ==========================================================================
#  ASK
# ==========================================================================

def ask(question, thread_id, label=None, fingerprint=None):
    print(f"[ASK] use_api={use_api()}", flush=True) 
    """One question, one answer, both modes identical.

    Returns the AskResponse shape — answer, tool, tier, confidence, result,
    expression, retries, latency_s — so the interface renders the same fields
    whichever transport produced them.

    `label` and `fingerprint` are ignored over HTTP: the server reads them from
    whatever IT has loaded, which is the correct source. They are parameters
    only because local mode has to build the graph state by hand, and asking
    the caller for them beats importing describe() twice.
    """
    if use_api():
        return _request(
            "POST",
            "/ask",
            timeout=ASK_TIMEOUT,
            json={"question": question, "thread_id": thread_id},
        )

    import time

    from src.tools.dataset import describe

    meta = describe()
    if meta is None:
        raise ApiError("No dataset is loaded.", status=404)

    state = {
        "question": question,
        "thread_id": thread_id,
        "fingerprint": fingerprint if fingerprint is not None else meta.get("fingerprint"),
        "label": label or meta["label"],
        "retry_count": 0,
    }
    config = {"configurable": {"thread_id": thread_id}}

    started = time.time()
    try:
        out = _local_graph().invoke(state, config)
    except Exception as exc:
        raise ApiError(f"{type(exc).__name__}: {exc}") from exc

    decision = out.get("decision") or {}
    raw = (out.get("execute_result") or {}).get("result")

    return {
        "answer": out.get("answer") or "No answer was produced.",
        "tool": decision.get("tool"),
        "tier": _TIER.get(decision.get("tool")),
        "confidence": decision.get("confidence"),
        "result": raw,
        "expression": raw.get("expression") if isinstance(raw, dict) else None,
        "retries": out.get("retry_count", 0),
        "thread_id": thread_id,
        "latency_s": round(time.time() - started, 2),
    }


# ==========================================================================
#  CONVERSATION
# ==========================================================================

def list_threads(limit=50):
    """Every stored conversation, most recent first."""
    if use_api():
        return _request("GET", "/threads", params={"limit": limit})["threads"]

    import sqlite3

    from src.store import get_connection

    connection = get_connection()
    try:
        rows = connection.execute(
            """
            SELECT thread_id, COUNT(*) AS n_turns, MIN(id) AS first_id,
                   MAX(created_at) AS last_active
            FROM turn GROUP BY thread_id ORDER BY MAX(id) DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()

        titles = {}
        if rows:
            ids = [row["first_id"] for row in rows]
            placeholders = ",".join("?" * len(ids))
            for row in connection.execute(
                f"SELECT id, question FROM turn WHERE id IN ({placeholders})", ids
            ).fetchall():
                titles[row["id"]] = row["question"]
    except sqlite3.Error:
        return []
    finally:
        connection.close()

    return [
        {
            "thread_id": row["thread_id"],
            "title": titles.get(row["first_id"], "(untitled)"),
            "n_turns": row["n_turns"],
            "last_active": row["last_active"],
        }
        for row in rows
    ]


def get_thread(thread_id, n=20):
    """The turns in one conversation, oldest first."""
    if use_api():
        return _request("GET", f"/threads/{thread_id}", params={"n": n})["turns"]

    from src.memory.conversation import recent_turns

    return recent_turns(thread_id, n)


# ==========================================================================
#  EXPLORE
# ==========================================================================

def get_columns(table="default"):
    """What can be charted in one table, split numeric / categorical."""
    if use_api():
        return _request("GET", "/columns", params={"table": table})

    from src.tools.dataset import _frame

    frame = _frame(table)
    return {
        "table": table,
        "numeric": frame.select_dtypes("number").columns.tolist(),
        "categorical": [
            column for column in frame.columns
            if frame[column].dtype == object
            or str(frame[column].dtype) == "category"
            or frame[column].nunique(dropna=True) <= 12
        ],
        "n_rows": int(len(frame)),
    }


def explore(**spec):
    """One chart's numbers.

    LOCAL MODE CALLS THE ENDPOINT FUNCTION DIRECTLY rather than reimplementing
    six chart computations. They are already written, they are pure, and a
    second copy would drift from the first — the box whiskers would end up
    using a different fence rule in one mode than the other, and nobody would
    notice until the charts disagreed.

    Its HTTPExceptions become ApiError so the caller still writes one except.
    """
    if use_api():
        return _request("POST", "/explore", timeout=QUICK_TIMEOUT, json=spec)

    from fastapi import HTTPException

    from src.api.main import explore as _explore
    from src.api.models import ExploreRequest

    try:
        return _explore(ExploreRequest(**spec))
    except HTTPException as exc:
        raise ApiError(str(exc.detail), status=exc.status_code) from exc


# ==========================================================================
#  OBSERVABILITY AND INTROSPECTION
# ==========================================================================

def counters(days=None):
    """Guardrail counts, optionally windowed."""
    if use_api():
        params = {"days": days} if days is not None else None
        return _request("GET", "/counters", params=params)

    from src.guardrails.counters import read_counts

    counts = read_counts(since_days=days)
    return {
        "window_days": days,
        "counts": counts,
        "total": sum(sum(actions.values()) for actions in counts.values()),
    }


def tools():
    """The eleven tools and what they take."""
    if use_api():
        return _request("GET", "/tools")

    from src.api.main import tools as _tools

    return _tools()


def vocabulary():
    """The router's view of the loaded data."""
    if use_api():
        return _request("GET", "/vocabulary")

    from src.tools.analytics import vocabulary as _vocabulary
    from src.tools.dataset import describe

    meta = describe()
    if meta is None:
        raise ApiError("No dataset is loaded.", status=404)
    return {"vocabulary": _vocabulary(), "dataset": meta}


# ==========================================================================
#  LOCAL MODE INTERNALS
# ==========================================================================

_TIER = {
    "aggregate_metric": 1, "compare_groups": 1, "crosstab_rate": 1,
    "band_distribution": 1,
    "predict_default": 2, "predict_churn": 2, "score_population": 2,
    "simulate_loan": 2, "get_segment_profile": 2, "get_feature_importance": 2,
    "answer_freeform": 3,
}

# Compiled once per process. Module-level rather than a Streamlit cache so the
# same client works from a script, a notebook or the MCP server without any of
# them knowing about st.cache_resource.
_GRAPH = None


def _local_graph():
    """The compiled graph, built on first use.

    Lazily, because HTTP mode never needs it and building it opens a sqlite
    connection and loads the whole graph module for nothing.
    """
    global _GRAPH
    print(f"[LOCAL_GRAPH] _GRAPH={'BUILT' if _GRAPH is not None else 'NONE'} id={id(_GRAPH)}", flush=True)
    if _GRAPH is None:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        from src.graph.build import build_graph

        connection = sqlite3.connect("graph_checkpoints.db", check_same_thread=False)
        _GRAPH = build_graph(checkpointer=SqliteSaver(connection))

    return _GRAPH 