"""
src/graph/test_execute.py
=========================

Three hand-built decisions, one per exit path that can fire on a real route.
No graph, no LLM — construct the RouterDecision directly and call execute.

Requires a loaded dataset, because the tools read _frame(). Run the upload
pipeline first, or point dataset.py at a fixture, so aggregate_metric has a
frame to work on.
"""

from src.graph.router import RouterDecision, Tool
from src.graph.execute import execute


def show(label, decision):
    out = execute(decision)
    print(f"\n--- {label} ---")
    print(f"ok        {out['ok']}")
    print(f"tool      {out['tool']}")
    print(f"retryable {out['retryable']}")
    print(f"error     {out['error']}")
    # result can be a large dict; print its type and a peek, not the whole thing
    r = out["result"]
    print(f"result    {type(r).__name__}: {str(r)[:120]}")


# 1. SUCCESS — real tool, valid params. ok True, retryable False, result is
#    aggregate_metric's dict.
good = RouterDecision(
    tool=Tool.AGGREGATE_METRIC,
    params={"metric": "defaulted"},
    confidence=0.98,
    reason="a straight overall rate",
)

# 2. ToolError — Karachi is not a region. validate() raises ToolError; execute
#    catches it. ok False, retryable TRUE, error names the real regions.
bad_value = RouterDecision(
    tool=Tool.AGGREGATE_METRIC,
    params={"metric": "defaulted", "filters": {"region": "Karachi"}},
    confidence=0.95,
    reason="rate for one region",
)

# 3. TypeError — `nonsense` is not a parameter of aggregate_metric, so **
#    raises TypeError, not ToolError. ok False, retryable FALSE.
bad_key = RouterDecision(
    tool=Tool.AGGREGATE_METRIC,
    params={"metric": "defaulted", "nonsense": 1},
    confidence=0.9,
    reason="a route that built a bad param key",
)

# 4. out_of_scope — no function runs. ok True, result is the reason text.
scope = RouterDecision(
    tool=Tool.OUT_OF_SCOPE,
    params={},
    confidence=0.96,
    reason="This system has no time dimension, so month-over-month is out of scope.",
)


if __name__ == "__main__":
    from src.tools.dataset import load_reference
    load_reference()          # fills _DATA with the reference parquets

    show("success", good)
    show("ToolError (retryable)", bad_value)
    show("TypeError (not retryable)", bad_key)
    show("out_of_scope", scope) 