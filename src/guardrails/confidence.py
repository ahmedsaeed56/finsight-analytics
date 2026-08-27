"""
src/guardrails/confidence.py
============================

Whether the router was sure enough to act on.

    RouterDecision  ->  proceed, proceed-and-log, clarify, or failed

CONTROL POINT 3 of five, and the last one to become buildable — it gates the
router's output, so it could not exist before the router did.

WHY A SEPARATE CHECK AT ALL
---------------------------
The model produces the confidence score itself, and a model asked to enforce
its own threshold will sometimes not. So the score is a PROPOSAL and this is
the DECISION: plain arithmetic against two constants, with no way to be talked
out of it.

Same division as everywhere else here. The router proposes a tool, validate()
decides whether its parameters are real. The model writes an expression,
observe() decides whether the result answers anything.

DETERMINISTIC OVERRIDES
-----------------------
Some routes are wrong at high confidence often enough that trusting the score
is not safe. The clearest case: `aggregate_metric` with `group_by` set and
`metric` absent — "what's the rate by region", "how do they compare per
purpose". The router keeps scoring these at 0.95 because a valid group_by
feels complete, but picking `defaulted` when the user meant `churned_12m` (or
the other way) produces a fluent, correctly-computed answer about a column
they never asked about. A beginner has no way to notice.

The rule can't live in the prompt alone — it does, and gets ignored. So it
lives here too, as arithmetic the model cannot argue with: metric missing on
a grouped call means clarify, regardless of the score.

Kept narrow. Every override is an escape from the score, and the score is
still the right signal in almost every other case. New ones need the same
justification: a well-documented failure mode the router keeps not catching.

PURE — DECIDES, DOES NOT LOG
----------------------------
The caller records. That keeps this runnable without a database, which is what
lets an eval set push thirty decisions through it without writing thirty rows.

Imports only the two thresholds. NOT RouterDecision — Python does not need the
type to read an attribute, and importing it would make src/guardrails depend on
src/graph, which is backwards: the graph uses this, not the reverse.
"""

from src.config import CONFIDENCE_CLARIFY, CONFIDENCE_PROCEED


def check_confidence(decision):
    """Decide what happens to a routing decision.

    Parameters
    ----------
    decision
        What route() returned. Only `failed`, `tool`, `params`, `confidence`
        and `reason` are read.

    Returns
    -------
    dict — allowed, action, message.

    `action` is one of four strings, and they are the values that land in the
    counter log's action column, so read_counts can tell them apart:

        proceed          the route is clear; run it
        proceed_logged   the tool is right, a parameter was a judgement call
        clarify          too uncertain to run; ask the user which they meant
        failed           the routing call itself did not complete

    `message` is None when proceeding, and user-facing text otherwise. It is
    the only part of this return a human ever sees — `action` is a signal for
    the graph's conditional edge, not an instruction for a model.
    """
    # FIRST, and it has to be. A failed route carries confidence 0.0, so any
    # other order sends it down the clarify branch and it never reaches its
    # own case — leaving "clarify: 15" in the log where "clarify: 12,
    # failed: 3" was the useful answer. Three API failures mean something is
    # down; twelve unclear questions mean nothing is.
    if decision.failed:
        return {
            "allowed": False,
            "action": "failed",
            # No "please rephrase" appended. Their question may have been
            # perfectly clear — the call broke, not the question. route()'s
            # fallback already ends with "Please try again."
            "message": decision.reason,
        }

    # DETERMINISTIC CLARIFY — a grouped call with no metric.
    #
    # Fires BEFORE the confidence branches, because this is the case the
    # score gets wrong: the router keeps returning 0.95 on "rate by region"
    # even though the metric was never named, because a valid group_by feels
    # like a complete question. It is not — "rate" could mean the default
    # rate or the churn rate, and picking one silently is exactly the failure
    # mode the prompt rule was meant to catch and does not.
    #
    # Narrow on purpose: only aggregate_metric, only when a group_by is set
    # AND metric is missing. Every other tool either requires a metric in its
    # signature (so validate() catches the omission) or does not take one
    # (band_distribution, score_population). And a metric with no group_by is
    # complete — it means the overall figure.
    if _needs_metric_clarification(decision):
        return {
            "allowed": False,
            "action": "clarify",
            # A statement, not a question — confidence.py used to append its
            # own "Which did you mean?" and produced two questions in a row.
            # Naming the two most common metric readings by hand rather than
            # asking the model to write the reason: the model is the reason
            # we are here, and its confidence said no clarification was
            # needed at all.
            "message": (
                f"The grouping by {decision.params['group_by']} is clear, but "
                f"which figure did you want per group? The default rate, the "
                f"churn rate, and the average loan amount are the common "
                f"readings, and each gives different numbers."
            ),
        }

    if decision.confidence >= CONFIDENCE_PROCEED:
        return {
            "allowed": True,
            "action": "proceed",
            "message": None,
        }

    if decision.confidence >= CONFIDENCE_CLARIFY:
        # The tool is right; something about the parameters was a judgement
        # call. Runs anyway — but logged, so a pattern of borderline routes on
        # one kind of question is visible rather than buried in the successes.
        return {
            "allowed": True,
            "action": "proceed_logged",
            "message": None,
        }

    # Below the floor. The router was guessing at the tool, or at a parameter
    # that changes the answer.
    #
    # A clarifying question costs one turn. A confidently wrong route costs
    # trust, because the user gets a real number computed correctly for a
    # question they did not ask, and nothing in the output says so.
    return {
        "allowed": False,
        "action": "clarify",
        # The reason IS the clarifying question. router.md tells the router to
        # write it as one — "Did you mean the churn rate or the default rate?"
        # — so appending "Which did you mean?" here produced two questions in a
        # row: "...or something else? Which did you mean?" One writer owns the
        # sentence, and it is the router, which knows what was ambiguous.
        "message": decision.reason,
    }


def _needs_metric_clarification(decision):
    """True when a grouped aggregate call has no metric named.

    Split out for readability and for testability — the deterministic override
    is worth being able to hit directly in a unit test without constructing a
    whole decision that would have proceeded.

    Guards each attribute access with .get() and hasattr() because a failed
    decision may not carry a full params dict, and this check runs before the
    confidence branches decide what to do with it.
    """
    if getattr(decision, "tool", None) != "aggregate_metric":
        return False

    params = getattr(decision, "params", None) or {}

    group_by_set = params.get("group_by") is not None
    metric_missing = params.get("metric") is None

    return group_by_set and metric_missing 