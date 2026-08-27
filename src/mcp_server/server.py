"""
src/mcp_server/server.py
========================

The JazzCash analytics system, exposed as MCP tools.

    Claude (client)  ->  this MCP server  ->  FastAPI  ->  the graph  ->  tools

WHAT THIS IS
------------
MCP — Model Context Protocol — is how an AI client (Claude Desktop, most
obviously) calls external functions. This server publishes four tools; when
Claude decides one is relevant to a conversation, it calls it and reads the
JSON back.

WHY IT ONLY WRAPS THE API
-------------------------
This file could import the graph directly, the way Streamlit's local mode does.
It deliberately does not. Going through FastAPI means:

  - the API is a real boundary rather than a demo. If it works for MCP it will
    work for anything else that speaks HTTP.
  - the guardrails, the router, the confidence gate and the narrator are all
    on the far side. Nothing about "this is Claude asking" gets special
    treatment.
  - one dataset, one graph, one truth. A second MCP server importing the graph
    would have its own _DATA, and Claude's answers would drift from what a
    Streamlit user was seeing.

WHY FOUR TOOLS, NOT ELEVEN
--------------------------
Exposing the eleven analytics tools directly would let Claude bypass the
router, which is the piece that makes this project interesting. The router
knows which tool a question needs; Claude does not, and asking Claude to route
into an unfamiliar tier system is inviting the mistake the router exists to
prevent.

So Claude gets one door — `ask` — and three read-only tools that let it orient
itself before knocking:

  ask              -> put a question through the whole pipeline
  get_dataset      -> what data the answers describe right now
  list_tools       -> the shape of what the system can answer
  get_vocabulary   -> the metrics, groupings and allowed values in the file

WHY STDIO
---------
MCP servers can talk over stdio (Claude Desktop launches this script and
pipes messages) or HTTP (a running service). Stdio is the right choice for a
portfolio demo: one command in claude_desktop_config.json and Claude can use
this. HTTP would need a hosted URL.

The API is a different question — that has to be running separately, because
this server calls it. See "the two processes" below.

THE TWO PROCESSES
-----------------
You now have three things:

    uvicorn        the FastAPI server on :8000
    this script    launched by Claude Desktop over stdio
    Claude Desktop the MCP client

The MCP server assumes the API is up. If it is not, `ask` returns an error
that names the fix ("start it with: python -m uvicorn ...") rather than a
cryptic connection refused. Same principle as the client module: a failure
should say what would work.

DO NOT PRINT
------------
Stdio is the transport. Anything this script writes to stdout that is NOT an
MCP message corrupts the protocol and Claude Desktop disconnects. Which means
no print() for debugging, no logging to stdout. The `mcp` library handles the
protocol traffic; keep everything else off stdout.
"""

from typing import Any, Optional

from mcp.server.mcpserver import MCPServer 

from src.api import client
from src.api.client import ApiError

# Force HTTP mode for this server. The environment might have JAZZCASH_USE_API
# unset, and defaulting to in-process would silently import the whole graph
# into the MCP process — which is exactly the coupling this file exists to
# avoid.
import os
os.environ["JAZZCASH_USE_API"] = "true"


mcp = MCPServer(
    name="jazzcash-analytics",
    instructions=(
        "Natural-language analysis over a loan book from JazzCash — a Pakistani "
        "mobile-money service. Every figure returned by this system is computed "
        "in Python by parameterised tools; the model routes questions and "
        "narrates results, and never calculates anything itself.\n\n"
        "Use `ask` for any question about the loan book: rates, comparisons, "
        "risk scores, ranked lists. It handles routing, guardrails, and "
        "narration internally — pass the user's question through unchanged.\n\n"
        "Before asking, check `get_dataset` to know which file is loaded. Use "
        "`list_tools` to see the shape of what can be asked. Use "
        "`get_vocabulary` if you need the exact metric names or allowed values "
        "in the current data.\n\n"
        "The system will refuse questions the data cannot answer rather than "
        "guessing — trust its refusals."
    ),
)


# ==========================================================================
#  TOOLS
# ==========================================================================

@mcp.tool()
def ask(question: str, thread_id: str = "mcp-default") -> dict[str, Any]:
    """Put one question through the full analytics pipeline.

    The system will:
      1. Refuse if the question is off-topic, injection, or empty.
      2. Rewrite follow-ups into standalone form using previous turns.
      3. Route to one of eleven tools — Tier 1 measures what happened in the
         file, Tier 2 gives a model's prediction, Tier 3 writes pandas at
         runtime.
      4. Ask for clarification if the router is not confident enough.
      5. Compute the figure in Python.
      6. Narrate the result, with caveats intact.

    Args:
        question: The question in plain language. Examples: "what is the
            default rate by region?", "how risky is loan L500042?", "which
            customers are most likely to churn?", "should we approve
            C100234 for a 50000 nano loan over 6 months?".
        thread_id: Reuse the same id across turns for a conversation the
            system remembers. Use a fresh id to start over.

    Returns:
        answer      the narrated reply, for a person to read
        tool        which tool produced the figures, or 'out_of_scope'
        tier        1, 2, or 3 — see the routing description above
        confidence  0 to 1; below 0.70 the system asked to clarify
        result      the tool's raw output, so figures can be quoted exactly
        expression  Tier 3 only: the pandas that produced the number
        retries     how many reroutes were spent recovering
        latency_s   wall-clock seconds

    A refusal is NOT a failure. Questions the data cannot answer come back with
    `tool: "out_of_scope"` and an explanation — treat that as the answer, do
    not retry with rewordings.
    """
    try:
        return client.ask(question, thread_id)
    except ApiError as exc:
        # Passed through as a dict rather than raising, so Claude sees the
        # message and can decide what to say about it. The API's error text is
        # already written for a person — it names what would work.
        return {
            "error": exc.detail,
            "status": exc.status,
            "hint": _api_down_hint() if exc.status is None else None,
        }


@mcp.tool()
def get_dataset() -> dict[str, Any]:
    """What data the answers currently describe.

    Every figure this system returns describes ONE dataset — a specific
    upload with a specific fingerprint. Check this before asking anything if
    you need to caveat an answer with the source.

    Returns:
        label        what the dataset is called
        fingerprint  content hash; null for the reference extract
        loaded_at    UTC timestamp of the load
        as_of        the freeze date derived from the file
        rows         count per feature table

    Returns null if nothing is loaded — in that case, `ask` will refuse until
    someone loads a file.
    """
    try:
        result = client.get_dataset()
        if result is None:
            return {"loaded": False, "message": "No dataset is loaded."}
        return {"loaded": True, **result}
    except ApiError as exc:
        return {"error": exc.detail, "hint": _api_down_hint()}


@mcp.tool()
def list_tools() -> dict[str, Any]:
    """The eleven tools the router can pick from, and what each takes.

    Use this to understand what the system can and cannot answer. You will
    never call these tools directly — `ask` routes to them — but knowing the
    surface helps you frame questions that match what exists.

    The tiers matter: Tier 1 is direct measurement, Tier 2 is model prediction,
    Tier 3 is generated pandas. A question that reads like Tier 1 but has no
    matching tool is refused, not approximated with Tier 3.
    """
    try:
        return client.tools()
    except ApiError as exc:
        return {"error": exc.detail, "hint": _api_down_hint()}


@mcp.tool()
def get_vocabulary() -> dict[str, Any]:
    """The metrics, group-by columns and allowed values in the LOADED file.

    Generated from the current data, not hardcoded — a file with a seventh
    region would show that region here. Use this to check that a term you plan
    to use in `ask` actually exists: "credit_score_band", "declared_income_band",
    "region" values like "Balochistan" or "Sindh", and so on.

    The system rejects near-miss values ("Karachi" is not a region — Sindh is),
    so it is worth checking the exact spelling here rather than being refused
    and retrying.
    """
    try:
        return client.vocabulary()
    except ApiError as exc:
        return {"error": exc.detail, "hint": _api_down_hint()}


# ==========================================================================
#  HELPERS
# ==========================================================================

def _api_down_hint():
    """One message for every 'API not reachable' case.

    Every tool can hit this and the fix is always the same, so it lives once
    rather than being written five times slightly differently.
    """
    return (
        "The FastAPI server is not running. Start it in a terminal:\n"
        "    python -m uvicorn src.api.main:app\n"
        "It listens on http://127.0.0.1:8000 — this MCP server calls it."
    )


# ==========================================================================
#  ENTRY POINT
# ==========================================================================

if __name__ == "__main__":
    # Default transport is stdio, which is what Claude Desktop launches this
    # over. FastMCP handles reading from stdin and writing to stdout —
    # anything else that writes to stdout will corrupt the protocol.
    mcp.run()  