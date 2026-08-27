"""
src/eval/run_cases.py
=====================

Push every case through the graph once and record what happened.

    CASES  ->  one run each  ->  eval_results.json  +  a readable table

STEP 2 OF FOUR. The cases fix the tool and the branch in advance; this run is
what supplies the VALUES nobody could know beforehand — a model's probability,
a generated expression, the exact phrasing of a refusal. You read the output,
decide case by case whether the system did the right thing, and only then does
a verified value become an assertion.

WRITES TO DISK, DELIBERATELY. Thirty cases is thirty real LLM calls and several
minutes. Re-running to re-read an answer is waste, so the results are saved and
the hand-check happens against the file.

ONE THREAD PER CASE. These are single-turn cases and must not resolve against
each other — a shared thread would let case 12 rewrite itself using case 11's
answer, which is exactly the leak the isolation test exists to catch. The run
timestamp is in every thread id so a second run cannot see the first one's turns.

CHECKS WHAT IT CAN, FLAGS THE REST. The tool, the branch, the retry count and
the forbidden phrases are decided here in Python. Whether an answer is WELL
PHRASED is left for your eyes — this prints it and stays quiet.
"""

import json
import time
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from src.eval.cases import CASES, ANSWERED, REFUSED, CLARIFIED, BLOCKED
from src.graph.build import build_graph
from src.memory.cache import clear_old
from src.tools.dataset import load_reference

# Beside the eval code, not in outputs — this is working material for the
# hand-check, not a deliverable.
_RESULTS_PATH = Path("src/eval/eval_results.json")


def infer_branch(out):
    """Which path the turn actually took, read back from the final state.

    The graph does not label its own exit, so the branch is reconstructed from
    what the state holds at the end:

        no decision at all      -> scope blocked it before the router ran
        decision but no result  -> the confidence gate ended the turn
        tool is out_of_scope    -> the router refused
        otherwise               -> a tool ran and was narrated

    Order matters: a blocked turn has no decision, and reading decision["tool"]
    on it would raise before the check that explains why it is empty.
    """
    decision = out.get("decision")

    if not decision:
        return BLOCKED

    if out.get("execute_result") is None:
        return CLARIFIED

    if decision.get("tool") == "out_of_scope":
        return REFUSED

    return ANSWERED


def check(case, out, latency):
    """Everything Python can decide without judgement.

    Returns a list of failure strings — empty means every mechanical check
    passed. A pass here is NOT a pass overall: the answer still has to be read.
    """
    failures = []
    answer = (out.get("answer") or "")
    lowered = answer.lower()

    # --- branch -------------------------------------------------------
    actual_branch = infer_branch(out)
    if actual_branch != case["branch"]:
        failures.append(f"branch: expected {case['branch']}, got {actual_branch}")

    # --- tool ---------------------------------------------------------
    decision = out.get("decision") or {}
    actual_tool = decision.get("tool")
    if case["tool"] is not None and actual_tool != case["tool"]:
        failures.append(f"tool: expected {case['tool']}, got {actual_tool}")

    # --- forbidden phrases --------------------------------------------
    # The narrator's worst failures have textual signatures. "will default" in
    # a Tier 2 answer states a model's estimate as a fact about the loan, and
    # that is catchable without a judge.
    for phrase in case.get("forbidden", []):
        if phrase.lower() in lowered:
            failures.append(f"forbidden phrase present: '{phrase}'")

    # --- required substrings ------------------------------------------
    # Filled in AFTER the hand-check for value cases; present from the start
    # for the ones knowable in advance (a caveat keyword, a region name).
    for phrase in case.get("must_contain", []):
        if phrase.lower() not in lowered:
            failures.append(f"missing expected content: '{phrase}'")

    # --- latency ------------------------------------------------------
    # Only asserted on the injection cases, where a fast block IS the proof
    # that no model was called.
    cap = case.get("max_latency_s")
    if cap is not None and latency > cap:
        failures.append(f"latency {latency:.2f}s over cap {cap}s — an LLM was probably called")

    # --- retry --------------------------------------------------------
    # retry_count is bumped by the retry node, so a non-zero value is direct
    # evidence the ToolError edge fired.
    expected_retry = case.get("expect_retry")
    actual_retry = out.get("retry_count", 0)
    if expected_retry is True and actual_retry == 0:
        failures.append("expected the retry edge to fire, it did not")
    if expected_retry is False and actual_retry > 0:
        failures.append(f"unexpected retry, count {actual_retry}")

    return failures


def run_one(app, case, run_id):
    """One case, one fresh thread, timed."""
    thread_id = f"eval-{run_id}-{case['id']}"

    state = {
        "question": case["question"],
        "thread_id": thread_id,
        "fingerprint": None,
        "label": "reference extract",
        "retry_count": 0,
    }
    config = {"configurable": {"thread_id": thread_id}}

    started = time.time()
    try:
        out = app.invoke(state, config)
        error = None
    except Exception as exc:
        # A crash is a RESULT, not a reason to stop the run. Twenty-nine other
        # cases still have something to say.
        out = {}
        error = f"{type(exc).__name__}: {exc}"
    latency = time.time() - started

    failures = check(case, out, latency) if error is None else [f"CRASHED — {error}"]
    decision = out.get("decision") or {}

    return {
        "id": case["id"],
        "question": case["question"],
        "note": case.get("note", ""),
        "expected_tool": case["tool"],
        "actual_tool": decision.get("tool"),
        "expected_branch": case["branch"],
        "actual_branch": infer_branch(out) if error is None else None,
        "confidence": decision.get("confidence"),
        "params": decision.get("params"),
        "retry_count": out.get("retry_count", 0),
        "answer": out.get("answer"),
        "latency_s": round(latency, 2),
        "failures": failures,
        "error": error,
    }


def main():
    load_reference()

    # A cold cache, so every case exercises the real path. Without this, a
    # second run reads answers the first one stored and proves nothing about
    # the tools.
    clear_old(days=0)

    run_id = str(int(time.time()))
    results = []

    with SqliteSaver.from_conn_string(":memory:") as saver:
        app = build_graph(checkpointer=saver)

        for i, case in enumerate(CASES, start=1):
            print(f"[{i:>2}/{len(CASES)}] {case['id']:<28} ", end="", flush=True)
            result = run_one(app, case, run_id)
            results.append(result)

            mark = "FAIL" if result["failures"] else "ok"
            print(f"{mark:<5} {result['latency_s']:>6.2f}s  {result['actual_tool']}")

    _RESULTS_PATH.write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )

    # --- summary ------------------------------------------------------
    failed = [r for r in results if r["failures"]]

    print(f"\n{'=' * 70}")
    print(f"{len(results) - len(failed)}/{len(results)} passed the mechanical checks")
    print(f"results written to {_RESULTS_PATH}")

    if failed:
        print(f"\nMECHANICAL FAILURES — {len(failed)}:")
        for r in failed:
            print(f"\n  {r['id']}")
            print(f"    Q: {r['question']}")
            for f in r["failures"]:
                print(f"    - {f}")

    # The point of the run. Every answer needs reading, including the ones
    # that passed — a mechanically-clean answer can still be wrong.
    print(f"\n{'=' * 70}")
    print("NOW READ EVERY ANSWER in eval_results.json. The checks above cover")
    print("routing and forbidden phrases; they cannot tell you whether a")
    print("number is right or a caveat is missing.")


if __name__ == "__main__":
    main() 