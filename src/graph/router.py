"""
src/graph/router.py
===================

The routing call. Question in, RouterDecision out.

    ROUTING = choose the tool + its parameters + a confidence + a reason.

Every question the graph answers passes through here. It is the ONE place
that has to know all four routing outputs at once — anywhere else that talks
about a tool talks about ONE tool. So the schema, the response type, and the
call are together, and every field the router controls is visible in this
file.

WHY structured_output RATHER THAN JSON_MODE
-------------------------------------------
`with_structured_output` binds a schema the API validates and returns as a
typed object. `response_mime_type="application/json"` returns a string that
parses. On any well-formed happy path both work; the interesting case is when
the model would otherwise return `{"tool": "aggregate_metric", "params": ...}`
followed by prose. json_mode leaves you writing a stripper; structured output
rejects the response before it reaches you.

The pin also removes the ability to hallucinate a tool name. `tool` is an
enum, and the API refuses anything outside it — so "use tool aggregate_metrics"
never reaches this file.

WHY THE ENUM IS COMPUTED, NOT WRITTEN OUT
------------------------------------------
`RouterResponse.tool` is `Tool | OutOfScope`. Adding a new tool to
`src/tools/` and registering it in `Tool` extends the router's permitted
outputs automatically — you don't come back to `router.py` to widen an
enum a second time. The tests catch that: 30/30 without touching this file.

TEMPERATURE = 0 (via the model)
--------------------------------
Confidence 0.90 with a temperature warming things up isn't confidence, it's
sampling noise. `src/llm.py` pins 0 and this file inherits that. The
"non-determinism" warnings in the docs are for creative writing.

CONFIDENCE IS A PROPOSAL, NOT A DECISION
----------------------------------------
Every RouterDecision carries a `confidence` field — the model's own estimate.
The router does not consult it. What that number means for control flow is
`src/guardrails/confidence.py`'s job, which is why that file exists as a
separate control point: the model produces a number, the guardrail decides
whether to trust it.

DETERMINISTIC METRIC-GUESS CHECK
--------------------------------
The router.md rule "grouping named but no metric = low confidence" gets
ignored — the model keeps returning `metric: "defaulted"` on questions like
"what's the rate by region?" at 0.98 confidence. Confidence.py cannot catch
it because by then `params["metric"]` is populated. So the check runs HERE,
against the raw user question, right after the LLM returns.

If aggregate_metric was picked, group_by is set, metric is set, AND the raw
question mentions none of that metric's names or synonyms — the router
guessed. Strip the metric and drop confidence to the clarify band so
confidence.py's existing branch produces the clarification. No new code
path, no override flag, no special counter category: it lands in the log as
an ordinary clarify.

Narrow on purpose. False negative (missing a synonym) costs one clarifying
turn; false positive (thinking they said "default" when they said "rate")
costs a wrong-column answer. The synonym list errs toward false negatives.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict

from pydantic import BaseModel, Field

from src.config import PROMPTS_DIR
from src.llm import get_model


class Tool(str, Enum):
    """Every tool the graph can call. Extend here to widen the router."""

    # Tier 1 — parameterised analytics.
    AGGREGATE_METRIC = "aggregate_metric"
    COMPARE_GROUPS = "compare_groups"
    CROSSTAB_RATE = "crosstab_rate"
    BAND_DISTRIBUTION = "band_distribution"

    # Tier 2 — model wrappers.
    PREDICT_DEFAULT = "predict_default"
    PREDICT_CHURN = "predict_churn"
    SCORE_POPULATION = "score_population"
    SIMULATE_LOAN = "simulate_loan"
    GET_SEGMENT_PROFILE = "get_segment_profile"
    GET_FEATURE_IMPORTANCE = "get_feature_importance"

    # Tier 3 — sandboxed generation.
    ANSWER_FREEFORM = "answer_freeform"

    # Not a tool; a valid decision. Kept on the enum so the schema accepts
    # it and there is only one authoritative list of legal `tool` values.
    OUT_OF_SCOPE = "out_of_scope"


class RouterDecision(BaseModel):
    """What the model returns. What the graph reads.

    `failed` is not a field the model sets — it is a marker the fallback path
    below uses to distinguish a real refusal (out_of_scope with a written
    reason) from a routing outage (out_of_scope because the call itself
    failed). The confidence guardrail reads it to choose the message.

    `confidence` is unclamped here on purpose. The model self-reports; if it
    returns 1.1 that is a signal to look at, not a validation error to hide.
    Clamping happens in the guardrail if it needs to.
    """

    tool: Tool = Field(description="Chosen tool or out_of_scope.")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the chosen tool. Empty for out_of_scope.",
    )
    confidence: float = Field(description="Self-reported, 0 to 1.")
    reason: str = Field(description="One sentence, statement form.")

    # Populated by the fallback path, never by the model. Kept out of the
    # bound schema below so the model cannot set it.
    failed: bool = Field(default=False, exclude=True)


# --------------------------------------------------------------------------
# METRIC-GUESS DETECTION
# --------------------------------------------------------------------------
# Words a user actually types when naming a metric. Every metric that a real
# question is likely to reach for gets an entry; a metric with no entry is
# not checked (assumed mentioned) rather than being incorrectly flagged.
#
# Kept synced with router.md's "Everyday words map onto columns" block —
# same synonyms, opposite direction. If either changes, update both.

_METRIC_SYNONYMS = {
    "defaulted": [
        "default", "defaulted", "bad loan", "went bad", "didn't pay",
        "non-performing", "npl",
    ],
    "churned_12m": [
        "churn", "churned", "left", "stopped using", "quit", "dropped",
        "attrition", "dormancy", "lapse",
    ],
    "amount_pkr": [
        "amount", "loan size", "loan amount", "how much they borrowed",
        "ticket size", "borrowed",
    ],
    "credit_score": ["credit score"],
    "age": ["age"],
    "total_txns": ["transaction", "txn"],
    "total_value": ["transaction value", "value moved"],
    "active_months": ["active months", "months active"],
    "savings_balance_pkr": ["savings", "savings balance"],
    "avg_monthly_inflow_pkr": ["inflow", "monthly inflow"],
    "inflow_to_loan_ratio": ["ratio", "loan-to-income", "dti", "affordability"],
    "complaints_12m": ["complaint"],
    "dependents": ["dependent"],
    "has_savings": ["savings"],
    "has_insurance": ["insurance"],
    "smartphone_user": ["smartphone"],
}


def _mentions_metric(question: str, metric: str) -> bool:
    """Did the user's raw text name this metric or one of its synonyms?

    Case-insensitive, word-boundary matched. Returns True conservatively —
    if the metric has no synonym list, treat it as mentioned.
    """
    if metric not in _METRIC_SYNONYMS:
        return True

    lowered = question.lower()
    for word in _METRIC_SYNONYMS[metric]:
        if re.search(r"\b" + re.escape(word) + r"\b", lowered):
            return True

    return False


def _apply_metric_guess_check(
    decision: RouterDecision, question: str
) -> RouterDecision:
    """Correct a silently-guessed metric on a grouped aggregate call.

    Fires only for `aggregate_metric` with `group_by` set. Everything else
    either requires the metric (validate() catches missing ones) or does
    not take one.

    When it fires: strip the metric, drop confidence into the clarify band,
    rewrite the reason. Confidence.py's existing branch produces the
    clarification — no new path, no override flag.
    """
    if decision.failed:
        return decision

    if decision.tool != Tool.AGGREGATE_METRIC:
        return decision

    params = decision.params or {}
    group_by = params.get("group_by")
    metric = params.get("metric")

    if group_by is None or metric is None:
        return decision

    if _mentions_metric(question, metric):
        return decision

    # Router guessed. Strip the metric, drop confidence below the clarify
    # threshold, replace the reason with a user-facing clarification.
    new_params = dict(params)
    new_params.pop("metric", None)

    return RouterDecision(
        tool=Tool.AGGREGATE_METRIC,
        params=new_params,
        confidence=0.50,
        reason=(
            f"The grouping by {group_by} is clear, but which figure did "
            f"you want per group? The default rate, the churn rate, and "
            f"the average loan amount are the common readings, and each "
            f"gives different numbers."
        ),
        failed=False,
    )


# The routing rules and the tool catalogue, read at import.
#
# Not the vocabulary — that comes with each dataset. If those files change
# the process needs a restart; that is the correct trade for keeping them
# out of the hot path.
_ROUTER_PROMPT = (PROMPTS_DIR / "router.md").read_text(encoding="utf-8")
_TOOLS_PROMPT = (PROMPTS_DIR / "tools.md").read_text(encoding="utf-8")

# The model configured for structured output, once. Keeping this at module
# level means the LangChain runnable's cache stays warm — reconstructing per
# call is measurable, and the router runs on the hot path of every question.
#
# `include_raw=False` returns the parsed object directly. `method="json_mode"`
# would return a string; the Gemini path here binds the schema.
_MODEL = get_model(light=True).with_structured_output(
    RouterDecision, include_raw=False
)


def route(question: str) -> RouterDecision:
    """One routing decision for one question.

    Parameters
    ----------
    question
        The resolved user question. Follow-ups have been rewritten upstream —
        this file does no history-tracking of its own.

    Returns
    -------
    RouterDecision — always. On API failure, `out_of_scope` at confidence 0
    with `failed=True` set, so the confidence guardrail can tell an outage
    from a real refusal.
    """
    system = _ROUTER_PROMPT + "\n\n" + _TOOLS_PROMPT

    try:
        decision = _MODEL.invoke([
            ("system", system),
            ("human", question),
        ])
    except Exception as exc:
        # Route around the failure with a shaped response the confidence
        # gate can read. Never raise from here — the graph has one edge for
        # "unroutable" and it goes through out_of_scope.
        return RouterDecision(
            tool=Tool.OUT_OF_SCOPE,
            params={},
            confidence=0.0,
            reason=(
                "The router couldn't complete this request. "
                f"({type(exc).__name__}) Please try again."
            ),
            failed=True,
        )

    # DETERMINISTIC OVERRIDE — runs after the LLM, before the caller sees it.
    # A silently-guessed metric on a grouped call becomes a clarification.
    decision = _apply_metric_guess_check(decision, question)

    return decision 