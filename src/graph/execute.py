"""
src/graph/execute.py
====================

Runs the tool the router chose.

    RouterDecision  ->  a result dict the narrator can read

router.py CHOOSES, this RUNS. It is the only file that imports every tool,
because it is the only place a tool name becomes a tool call.

FOUR EXITS, ONE SHAPE
---------------------
out_of_scope   ok=True,  result is the refusal text, nothing ran
success        ok=True,  result is the tool's dict
ToolError      ok=False, a parameter was rejected — RETRYABLE, the message
               names the valid values and a reroute can fix it
TypeError      ok=False, the router built params with a key no signature takes
               — NOT retryable, a second identical route repeats the mistake

Every path returns all five keys, so the narrator reads one shape and the
retry edge always finds `retryable`.
"""

from src.tools.analytics import (
    aggregate_metric,
    compare_groups,
    crosstab_rate,
    band_distribution,
)
from src.tools.inference import (
    predict_default,
    predict_churn,
    score_population,
    simulate_loan,
    get_segment_profile,
    get_feature_importance,
)
from src.tools.freeform import answer_freeform
from src.graph.router import Tool
from src.tools.errors import ToolError


# Enum member -> the function itself. No parentheses: the value is the function
# object, and () would call it at import. OUT_OF_SCOPE is absent on purpose —
# there is no function to run, and execute() handles it before the lookup.
_TOOLS = {
    Tool.AGGREGATE_METRIC: aggregate_metric,
    Tool.COMPARE_GROUPS: compare_groups,
    Tool.CROSSTAB_RATE: crosstab_rate,
    Tool.BAND_DISTRIBUTION: band_distribution,
    Tool.PREDICT_DEFAULT: predict_default,
    Tool.PREDICT_CHURN: predict_churn,
    Tool.SCORE_POPULATION: score_population,
    Tool.SIMULATE_LOAN: simulate_loan,
    Tool.GET_SEGMENT_PROFILE: get_segment_profile,
    Tool.GET_FEATURE_IMPORTANCE: get_feature_importance,
    Tool.ANSWER_FREEFORM: answer_freeform,
}


def _result(ok, tool, result=None, error=None, retryable=False):
    """Build the one return shape, so all four exits carry the same keys.

    The defaults mean each branch passes only what is true on its path, and
    every returned dict still holds ok, tool, result, error and retryable.
    """
    return {
        "ok": ok,
        "tool": tool,
        "result": result,
        "error": error,
        "retryable": retryable,
    }


def execute(decision):
    """Run the tool the router chose and return a single result shape.

    Parameters
    ----------
    decision
        A RouterDecision that already passed the confidence gate. Only
        `tool`, `params` and `reason` are read.

    Returns
    -------
    dict — ok, tool, result, error, retryable.
    """
    # out_of_scope first: there is no function to look up. Not a failure —
    # an honest refusal, so ok is True and the router's reason is the answer.
    if decision.tool is Tool.OUT_OF_SCOPE:
        return _result(True, decision.tool, result=decision.reason)

    # The enum guarantees this key exists — a name outside the enum was
    # unrepresentable at the router. No .get() with a fallback needed.
    fn = _TOOLS[decision.tool]

    try:
        result = fn(**decision.params)
        return _result(True, decision.tool, result=result)

    except ToolError as e:
        # Control point 1 firing: validate() rejected a parameter the router
        # proposed. The message names the bad value and lists the valid ones,
        # so a reroute can fix it — RETRYABLE. The graph node logs this.
        return _result(False, decision.tool, error=str(e), retryable=True)

    except TypeError as e:
        # `**` hit a key the function does not accept — a router mistake, not
        # a bad value. A second identical route makes the same mistake, so
        # this is NOT retryable and is worth distinguishing in the log.
        return _result(False, decision.tool, error=str(e)) 