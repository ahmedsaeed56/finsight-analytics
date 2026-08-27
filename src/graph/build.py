"""
src/graph/build.py
==================

The graph. Wires the nodes into one runnable pipeline.

    question  ->  scope  ->  rewrite  ->  route  ->  confidence
              ->  cache_get  ->  execute  ->  narrate  ->  cache_set
              ->  add_turn  ->  END

Every node already exists and was tested alone. This file does two things and
no more: it defines the STATE those nodes read and write, and it draws the
EDGES between them — including the two conditional ones that carry the real
control flow.

WHY EACH NODE IS A THIN WRAPPER
-------------------------------
The real functions have different signatures — route(question),
check_confidence(decision), narrate(question, result, label). A LangGraph node
does not: every node takes the STATE and returns a partial update to it. So
each wrapper pulls what its function needs out of state, calls it, and puts the
result back. The logic stays in the tested functions; these wrappers are
plumbing.

THE TWO EDGES THAT MATTER
-------------------------
after_confidence  proceed / clarify / failed. Only proceed continues; the
                  other two end the turn with a message.
after_execute     a ToolError is retryable — route again, guarded by a counter
                  so it cannot loop forever. Anything else goes to narrate.

Everything else is a straight line.

WHY tool IS STORED AS A STRING
------------------------------
The decision's `tool` is a Tool enum, but the checkpointer serialises state
with msgpack, which does not know that type — it warns now and will refuse
later. So route_node stores `tool` as its plain string value, and the one
place that needs the enum back (the cache key's tier lookup) reconstructs it
with Tool(...). State stays serialisable; the enum lives only where it is used.

THE MEMORY LOOP CLOSES AT add_turn
----------------------------------
rewrite reads past turns to resolve a follow-up, but only if something wrote
them. add_turn is that write: after every answered turn it records the RESOLVED
question and the answer against the thread, so the NEXT turn's rewrite has
history to resolve against. Without it, "what about by region?" routes as if it
stood alone.

WHY scope_node RESETS THE STATE
-------------------------------
The checkpointer restores the WHOLE of last turn's state before this turn runs.
Identity (thread_id, fingerprint, label) is meant to persist; the working
fields — answer, execute_result, decision, resolved_question, retry_count — are
this-turn scratch and must NOT. If last turn's `answer` survives, after_scope
sees it and short-circuits every follow-up straight to END, returning the stale
answer. So the entry node clears them.

WHERE THE COUNTERS ARE RECORDED
-------------------------------
Here, in the nodes — not inside the guardrails, which stay pure so the eval set
can run them without a database. freeform is the exception: it logs its own two
control points, because they fire deep inside a loop with no node beside them.
"""

from __future__ import annotations

from typing import Optional, TypedDict

from langgraph.graph import StateGraph, END

from src.graph.router import route,Tool
from src.guardrails.confidence import check_confidence
from src.guardrails.numbers import verify_numbers 
from src.graph.execute import execute
from src.graph.narrate import narrate
from src.guardrails.scope import check_scope
from src.guardrails.confidence import check_confidence
from src.guardrails.counters import record
from src.memory.rewrite import rewrite_question
from src.memory.cache import cache_get, cache_set
from src.memory.conversation import start_conversation, add_turn

# Beyond this many reroutes on one question, stop. A ToolError that survives
# two rewrites is one the router cannot fix — a value the data does not have,
# phrased so the router keeps proposing it. The cap turns an infinite loop into
# an honest failure the user can read.
MAX_RETRIES = 2

# Which tier a tool belongs to, for the cache key — the decision carries the
# tool but not the tier, and cache_get needs both. Keyed by the string value
# now, because state holds the tool as a string. Derived from the enum grouping
# rather than stored on the decision: the router should not have to know about
# caching to route.
_TIER = {
    Tool.AGGREGATE_METRIC.value: 1,
    Tool.COMPARE_GROUPS.value: 1,
    Tool.CROSSTAB_RATE.value: 1,
    Tool.BAND_DISTRIBUTION.value: 1,
    Tool.PREDICT_DEFAULT.value: 2,
    Tool.PREDICT_CHURN.value: 2,
    Tool.SCORE_POPULATION.value: 2,
    Tool.SIMULATE_LOAN.value: 2,
    Tool.GET_SEGMENT_PROFILE.value: 2,
    Tool.GET_FEATURE_IMPORTANCE.value: 2,
    Tool.ANSWER_FREEFORM.value: 3,
}

_OUT_OF_SCOPE = Tool.OUT_OF_SCOPE.value


class State(TypedDict, total=False):
    """What flows through the graph.

    total=False so a node may return only the keys it changed — LangGraph
    merges each partial update into the running state, and no node has to
    restate fields it did not touch.
    """

    # Identity, set by the caller before the run and carried throughout.
    thread_id: str
    fingerprint: Optional[str]
    label: Optional[str]

    # The question, raw then resolved.
    question: str
    resolved_question: Optional[str]

    # The router's decision, as a dict with `tool` as a plain string so the
    # checkpointer can serialise it.
    decision: Optional[dict]

    # execute()'s five-key result.
    execute_result: Optional[dict]

    # How many times this question has been rerouted after a ToolError.
    retry_count: int

    cache_hit: bool 

    force_escalate: bool

    # The final answer the user reads — set by whichever node ends the turn:
    # a guardrail refusal, or the narrator.
    answer: Optional[str]


# --------------------------------------------------------------------------
# NODES — each pulls from state, calls a tested function, returns an update
# --------------------------------------------------------------------------

def scope_node(state: State) -> dict:
    """Guardrail 2 — block direct injection and empty questions.

    Also the entry node, so it does two housekeeping jobs. It registers the
    thread (start_conversation is idempotent, so calling it every turn is
    harmless). And it RESETS the per-turn working fields — see the module
    docstring: without this, last turn's `answer` survives in the restored
    state and after_scope short-circuits every follow-up to END.

    A block ends the turn with a plain refusal; the internal reason is logged,
    never shown.
    """
    start_conversation(state["thread_id"], state.get("fingerprint"))

    verdict = check_scope(state["question"])

    if not verdict["allowed"]:
        record("scope", state["question"], verdict["code"], verdict["reason"])
        return {
            "answer": (
                "That question can't be processed. Please ask about the "
                "loaded data — its loans, customers, or transactions."
            ),
        }

    # Clear last turn's scratch. Identity fields (thread_id, fingerprint,
    # label) are deliberately NOT touched — they are meant to persist.
    return {
        "answer": None,
        "execute_result": None,
        "resolved_question": None,
        "decision": None,
        "retry_count": 0,
    }


def rewrite_node(state: State) -> dict:
    """Resolve a follow-up into a standalone question."""
    result = rewrite_question(
        state["question"],
        state["thread_id"],
        fingerprint=state.get("fingerprint"),
    )
    return {"resolved_question": result["question"]}


def route_node(state: State) -> dict:
    """Route the question. On escalation, force answer_freeform.

    force_escalate is set by retry_node after the last retry — at that point
    the router has already failed twice at lower tiers, so Tier 3 is the only
    move that can succeed. Override happens here rather than in the router
    itself because the router is a pure function of the question; escalation
    is a graph-state decision.
    """
    decision = route(state["resolved_question"]).model_dump()

    if hasattr(decision["tool"], "value"):
        decision["tool"] = decision["tool"].value 

    if state.get("force_escalate"):
        decision["tool"] = "answer_freeform"
        decision["params"] = {
            "question": state["resolved_question"],
            "table": "default",
        }
        decision["confidence"] = 0.75
        decision["reason"] = (
            "Escalating to a custom calculation after the parameterised "
            "tools couldn't express this."
        )

    return {"decision": decision} 


def confidence_node(state: State) -> dict:
    """Guardrail 3 — was the route sure enough to act on?

    A clarify or failed verdict ends the turn with its message; proceed and
    proceed_logged continue, the latter logged.
    """
    decision = _decision_obj(state["decision"])
    verdict = check_confidence(decision)

    # proceed_logged, clarify and failed all get a counter row; proceed does
    # not. The action string is the counter's own column value.
    if verdict["action"] != "proceed":
        record(
            "confidence",
            state["resolved_question"],
            verdict["action"],
            decision.reason,
        )

    if not verdict["allowed"]:
        return {"answer": verdict["message"]}

    return {}


def cache_get_node(state: State) -> dict:
    """Return a stored answer for this exact call, if one exists.

    Runs AFTER the router, because the key needs the tool and params the router
    produced. out_of_scope is never cached — there is no tool result to store.

    On a hit, the stored value is a tool result dict — but the user needs a
    SENTENCE, so a hit still narrates. What the cache saves is execute, which
    for Tier 3 is a paid LLM call. So a hit writes the result into
    execute_result in the shape execute would have produced, and lets the
    normal narrate step run.

    ALSO SETS cache_hit — cache_set_node reads it to skip re-writing the row
    it just read. Without the flag, every hit refreshes its own written_at
    for no reason: harmless data, wasted write.
    """
    decision = state["decision"]

    if decision["tool"] == _OUT_OF_SCOPE:
        return {}

    cached = cache_get(
        state.get("fingerprint"),
        _TIER[decision["tool"]],
        decision["tool"],
        decision["params"],
    )

    if cached is None:
        return {}

    return {
        "execute_result": {
            "ok": True,
            "tool": decision["tool"],
            "result": cached,
            "error": None,
            "retryable": False,
        },
        "cache_hit": True,
    }


def execute_node(state: State) -> dict:
    """Run the chosen tool. Skipped implicitly on a cache hit.

    If cache_get already filled execute_result, there is nothing to do. On a
    ToolError, the counter is recorded here (execute itself stays pure) before
    the edge decides whether to reroute.
    """
    if state.get("execute_result") is not None:
        return {}

    decision = _decision_obj(state["decision"])
    result = execute(decision)

    # Control point 1 — validate() rejected a parameter. Logged here because
    # execute stays pure and the graph is the caller that records.
    if not result["ok"] and result["error"]:
        action = "retry" if result["retryable"] else "failed"
        record("validate", state["resolved_question"], action, result["error"])

    return {"execute_result": result}

def narrate_node(state: State) -> dict:
    """Phrase the result as an answer the user reads, then verify numbers.

    CONTROL POINT 6 — every number in the narrated answer must appear in
    the result dict, or the answer is flagged. The check is a string search,
    not an LLM judge — plain arithmetic against the tool's own output.

    On a mismatch, log the counter and prepend a banner to the answer. The
    numbers are not silently corrected: the model got them wrong once, and a
    silent re-narrate might get them wrong again in a different way. A
    visible flag lets the user cross-check with the raw result.
    """
    answer = narrate(
        state["resolved_question"],
        state["execute_result"],
        label=state.get("label"),
    )

    result = state["execute_result"].get("result")
    check = verify_numbers(answer, result)

    if not check["ok"]:
        record("numbers", state["resolved_question"], "flagged",
               f"missing from result: {', '.join(check['missing'])}")
        banner = (
            "**One or more figures below don't appear in the underlying result — "
            "double-check before quoting.**\n\n"
        )
        answer = banner + answer

    return {"answer": answer} 


def cache_set_node(state: State) -> dict:
    """Store a fresh successful result, so the same call is free next time.

    Only stores real tool successes: a failure is not worth caching, and
    out_of_scope has no result. A cache HIT is also skipped — the row already
    exists and re-writing it just refreshes written_at for no reason.

    Best-effort — a cache that cannot write is slow, not broken.
    """
    if state.get("cache_hit"):
        return {}

    decision = state["decision"]
    result = state["execute_result"]

    if decision["tool"] == _OUT_OF_SCOPE:
        return {}
    if not result["ok"]:
        return {}

    cache_set(
        state.get("fingerprint"),
        _TIER[decision["tool"]],
        decision["tool"],
        result["result"],
        decision["params"],
    )
    return {} 


def add_turn_node(state: State) -> dict:
    """Record this exchange, so the next turn's rewrite has history.

    Stores the RESOLVED question, never the raw follow-up — storing "what about
    by region?" would make the next rewrite resolve against something already
    ambiguous. The tool and the turn's fingerprint go in too: the tool for
    tracing, the fingerprint so a mid-thread re-upload is visible as a
    divergence. Best-effort; a failed log must not break a delivered answer.
    """
    decision = state.get("decision") or {}
    add_turn(
        state["thread_id"],
        state["resolved_question"],
        state["answer"],
        tool=decision.get("tool"),
        fingerprint=state.get("fingerprint"),
    )
    return {}


# --------------------------------------------------------------------------
# EDGES
# --------------------------------------------------------------------------

def after_scope(state: State) -> str:
    """Blocked -> end. Allowed -> rewrite."""
    return "blocked" if state.get("answer") else "ok"


def after_confidence(state: State) -> str:
    """A clarify/failed verdict wrote an answer -> end. Otherwise proceed."""
    return "blocked" if state.get("answer") else "ok"


def after_execute(state: State) -> str:
    """The retry edge.

    A retryable ToolError routes back to the router for a second attempt — but
    only while under the cap. The reroute is a graph edge rather than a loop
    inside execute so it SHOWS in a trace: one visible pass through the router,
    not a silent retry.
    """
    result = state["execute_result"]

    if (
        not result["ok"]
        and result["retryable"]
        and state.get("retry_count", 0) < MAX_RETRIES
    ):
        return "retry"

    return "narrate"


def retry_node(state: State) -> dict:
    """Bump the counter and clear the stale result before rerouting.

    execute_result is cleared so execute_node's 'already populated' guard does
    not skip the re-run.

    ON THE SECOND RETRY, force the router to escalate to answer_freeform on
    the next pass. Two failures at the same tier mean the router cannot fix
    it by re-picking — the column doesn't exist in the whitelist, the value
    can't be expressed as a filter, whatever. Escalating to Tier 3 lets the
    sandbox try with real pandas, which is often the only route that can
    succeed. Without this, the third attempt fails identically to the first
    two and the turn dies having answered nothing.
    """
    new_count = state.get("retry_count", 0) + 1
    update = {
        "retry_count": new_count,
        "execute_result": None,
    }

    if new_count >= MAX_RETRIES:
        update["force_escalate"] = True

    return update 

# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------

def _decision_obj(decision_dict: dict):
    """Rebuild a RouterDecision from the dict stored in state.

    check_confidence and execute both read attributes off a decision object,
    while the state holds a dict (with tool as a string) for the checkpointer's
    sake. RouterDecision's `tool` field is a Tool enum, and pydantic coerces
    the string back to the enum on construction — so execute's _TOOLS lookup
    gets a real enum member, unchanged.
    """
    from src.graph.router import RouterDecision

    return RouterDecision(**decision_dict)


# --------------------------------------------------------------------------
# BUILD
# --------------------------------------------------------------------------

def build_graph(checkpointer=None):
    """Assemble the graph. Pass a checkpointer to persist state across turns.

    The checkpointer saves state per thread_id, so a follow-up in the same
    thread sees the previous turns. Without one the graph still runs, but every
    turn starts blank — fine for a single-shot test, wrong for a conversation.
    """
    g = StateGraph(State)

    g.add_node("scope", scope_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("route", route_node)
    g.add_node("confidence", confidence_node)
    g.add_node("cache_get", cache_get_node)
    g.add_node("execute", execute_node)
    g.add_node("retry", retry_node)
    g.add_node("narrate", narrate_node)
    g.add_node("cache_set", cache_set_node)
    g.add_node("add_turn", add_turn_node)

    g.set_entry_point("scope")

    g.add_conditional_edges("scope", after_scope, {"blocked": END, "ok": "rewrite"})

    g.add_edge("rewrite", "route")
    g.add_edge("route", "confidence")

    g.add_conditional_edges(
        "confidence", after_confidence, {"blocked": END, "ok": "cache_get"}
    )

    g.add_edge("cache_get", "execute")

    g.add_conditional_edges(
        "execute", after_execute, {"retry": "retry", "narrate": "narrate"}
    )

    g.add_edge("retry", "route")

    g.add_edge("narrate", "cache_set")
    g.add_edge("cache_set", "add_turn")
    g.add_edge("add_turn", END)

    return g.compile(checkpointer=checkpointer) 