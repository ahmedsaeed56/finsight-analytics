"""
src/tools/freeform.py
=====================

Tier 3 — questions the parameterised tools cannot express.

Tier 1 answers what its whitelists allow. Tier 2 answers what the models
predict. Anything else — a custom band, an unusual filter combination, a
correlation nobody anticipated — reaches here, where an LLM writes a pandas
expression and Python runs it.

THIS IS THE ONLY TIER THAT EXECUTES CODE IT DID NOT WRITE.
Everything above is Python that shipped in this repo. Here the expression
arrives at runtime from a model that read a user's question, so it is
untrusted by construction and runs sandboxed.

TWO OF THE FIVE CONTROL POINTS LIVE HERE
----------------------------------------
The sandbox and the iteration cap both fire in this file, and both log from
answer_freeform rather than from where they fire. run_sandboxed does not know
the question — it receives an expression — and the counter table requires one.
Threading a question through every tool signature for logging alone is the
wrong direction, so the logging happens in the one function that has both the
question and the outcome.
"""

import re
from pathlib import Path

import pandas as pd

from src.config import PROMPTS_DIR
from src.guardrails.counters import record
from src.llm import get_model
from src.tools.dataset import _frame
from src.tools.errors import ToolError

# Above this, listing a column's values is noise rather than help. Twelve
# covers every categorical in this schema (income band has 5, region 6,
# purpose 4, term_months 4) while excluding credit_score, age and every
# continuous column, which have hundreds.
MAX_LISTED_VALUES = 12

# Two attempts, not more. A third rarely differs from the second — by then the
# model has either understood the correction or is guessing — and every extra
# attempt is a paid call and a slower answer for a question that may simply
# not be answerable.
MAX_ATTEMPTS = 2

# The model's own refusal, and a first-class answer rather than a failure.
# Returned when the columns genuinely cannot support the question, which is a
# better outcome than a plausible substitute nobody can tell apart.
REFUSAL_PREFIX = "CANNOT_ANSWER:"

# The rules never change; the columns change with every upload. So the static
# half is a file and the dynamic half is generated per call.
_RULES_PATH = Path(PROMPTS_DIR) / "freeform.md"

# Substrings that never appear in a legitimate pandas expression and are the
# route to everything dangerous. Checked BEFORE eval, so a hit costs nothing
# and nothing runs.
#
# `__` matters as much as `import`: Python's escape routes do not need an
# import statement. ().__class__.__base__.__subclasses__() walks from an empty
# tuple to every loaded class, file handlers included.
BLOCKED_SUBSTRINGS = (
    "import",
    "__",
    "open(",
    "exec",
    "eval",
    "compile",
    "globals",
    "locals",
    "getattr",
    "setattr",
    "delattr",
    "vars(",
    "input(",
    "breakpoint",
)

# Errors worth a second attempt, because the fault is in the expression rather
# than in the data. A KeyError means the column does not exist, and asking
# again produces the same KeyError — see observe().
RETRYABLE_ERRORS = (
    "SyntaxError",
    "TypeError",
    "AttributeError",
    "ValueError",
    "IndexError",
)

# Beyond this a result is a dump rather than an answer. It goes into a context
# window, and a narrator handed 5,000 rows will pick something arbitrary out of
# the middle and present it as a pattern.
MAX_RESULT_ROWS = 200

# Question phrasing implies an answer shape. Deterministic keywords rather than
# a second model call: a judge that is itself an LLM adds a failure mode to
# catch a failure mode, and this project's rule is that Python judges.
#
# ORDER MATTERS. "How many loans in each region" contains both a scalar marker
# and a grouping one, and the grouping wins — so grouped is tested first.
_GROUPED_MARKERS = (
    " by ", " per ", "each ", "breakdown", "across ", "compare",
    "between ", "grouped", "for every",
)

_ROWS_MARKERS = (
    "top ", "list ", "show me", "which loan", "which customer",
    "largest", "smallest", "highest ", "lowest ", "name the",
)

_SCALAR_MARKERS = (
    "how many", "how much", "what share", "what proportion",
    "what percentage", "average ", "median ", "total ", "overall",
)

# What one row IS, and what the transaction columns COVER.
#
# The second half is why this cannot be inferred from the columns. total_txns,
# total_value and active_months appear in TWO frames and mean different things
# in each — pre-loan months in the loans frame, the full panel in the customer
# one. A summary that lists the name without the window lets an expression
# compare two quantities that are not the same measurement.
#
# The grain cannot be inferred either: churn and segment are both one row per
# customer and carry the same id column, so nothing in the columns tells them
# apart. The router already chose the frame, so it passes the name.
_FRAME_NOTES = {
    "default": (
        "one row per loan",
        "Transaction columns (total_txns, total_value, active_months, and the "
        "average_* and active_ratio columns derived from them) cover only the "
        "months BEFORE that loan's disbursed_date, so the window differs per "
        "row — months_available says how long it was. Compare loans using the "
        "average_* columns rather than the totals, which grow with the window.",
    ),
    "churn": (
        "one row per customer",
        "Transaction columns (total_counts, total_amount, active_months, "
        "first, last, difference) cover only the FIRST SIX MONTHS of the "
        "panel; the remaining months are the outcome period and are "
        "deliberately excluded, so no column here describes them. That "
        "six-month window is split in half: `first` counts transactions in "
        "its first three months and `last` in its second three, so "
        "`difference` is last minus first — negative means activity declined "
        "across the window. active_months therefore runs 0 to 6, not 0 to 12.",
    ),
    "segment": (
        "one row per customer",
        "Transaction columns (total_txns, total_value, active_months) cover "
        "the FULL panel — unlike the identically named columns in the loans "
        "frame, which cover pre-loan months only. Nothing is predicted here, "
        "so no window rule applies.",
    ),
}


def _values_of(series):
    """The allowed values, or None when there are too many to be useful.

    Listing them is what stops an expression filtering on "punjab" when the
    column holds "Punjab" — a filter that matches nothing, returns an empty
    result, and reads like a finding rather than a typo.

    dropna() because a null is not a value anything can filter on, and
    sorted() so the same frame produces the same summary on every run.
    """
    values = series.dropna().unique()

    if len(values) > MAX_LISTED_VALUES:
        return None

    # Categories sort in their DEFINED order, which for declared_income_band
    # is the meaningful one (<25k .. 250k+) rather than alphabetical.
    if isinstance(series.dtype, pd.CategoricalDtype):
        return [str(v) for v in series.cat.categories if v in set(values)]

    return [str(v) for v in sorted(values)]


def schema_summary(df, table=None):
    """A compact column reference for the expression generator.

    Generated from the LOADED frame, never from a stored document. The columns
    come from whatever the user uploaded, and a hardcoded list would drift the
    same way a stored row count does.

    Deliberately not prompts/schema.md. That file explains WHY columns exist,
    in prose, about the reference extract. This is a lookup table: what is
    here, what type it is, and what values it accepts.

    Parameters
    ----------
    df
        The frame the expression will run against.
    table
        Its name — "default", "churn" or "segment". Passed rather than
        guessed: churn and segment are indistinguishable by their columns, and
        the difference between them is exactly what has to be stated. Omitting
        it still produces a usable summary, minus the window note.

    Returns
    -------
    A plain string. It goes into a prompt, and a prompt is text.
    """
    grain, window = _FRAME_NOTES.get(table, ("one row per record", None))

    lines = [f"FRAME: {len(df):,} rows, {grain}"]
    if window:
        lines.append(window)
    lines.append("")

    # The longest name sets the column width, so the dtypes line up and the
    # summary reads as a table rather than a paragraph.
    width = max(len(str(c)) for c in df.columns)

    for column in df.columns:
        dtype = df[column].dtype
        line = f"{str(column):<{width}}  {dtype}"

        values = _values_of(df[column])
        if values is not None:
            line += "  ->  " + ", ".join(values)

        lines.append(line)

    return "\n".join(lines)


def run_sandboxed(expression, df):
    """Evaluate one generated pandas expression against one frame.

    CONTROL POINT 4 of five. Does not log — it has no question to log with.
    answer_freeform records the block.

    Returns
    -------
    (ok, payload)
        ok=True  -> payload is the raw pandas result
        ok=False -> payload is a dict: error_type, message, and blocked

    Catches rather than propagates, for three reasons. The loop needs the
    error as DATA, to put in the next prompt. observe() needs the error TYPE
    to decide whether a retry can help. And hostile input can raise things
    nobody anticipated — RecursionError, MemoryError — so one function owns
    "anything can happen" instead of every caller.

    HOW SAFE THIS IS, HONESTLY
    --------------------------
    Two layers: a substring blocklist, then eval with an emptied __builtins__
    and a namespace holding only `df` and `pd`. That stops the obvious
    attacks — open(), __import__(), reading the filesystem.

    It is NOT a real security boundary. Python has escape routes through
    attribute chains that need no import at all, which is why "__" is
    blocked as aggressively as "import". A determined attacker gets out.

    It also cannot stop an expression that never finishes. `while` is not
    valid in an expression, but a large enough cross-product can hang the
    process, and no namespace restriction catches that.

    The production path is a subprocess with a timeout — which fixes the hang
    and contains an escape in one move — and a container or a service like
    E2B if this ever serves more than one company's data.
    """
    lowered = expression.lower()

    for pattern in BLOCKED_SUBSTRINGS:
        if pattern in lowered:
            return False, {
                "error_type": "BlockedExpression",
                "message": (
                    f"the expression contains '{pattern}', which is not "
                    f"permitted. This tool evaluates pandas against the "
                    f"loaded frame only — it has no file, module or system "
                    f"access."
                ),
                # A refusal, not a mistake. observe() must not feed this back
                # and invite a second attempt at the same forbidden thing.
                "blocked": True,
                # Named so the log records WHICH pattern fired, not just that
                # something did — the counts are only useful if they can be
                # broken down.
                "pattern": pattern,
            }

        # __builtins__ emptied to strip the dangerous names (open, __import__,
    # exec, compile) that eval would otherwise expose. A small whitelist of
    # harmless callables is put back — len(df[...]), sum(series), and the
    # comparison helpers are all legitimate on pandas results and blocking
    # them just makes routine questions fail.
    safe_builtins = {
        "len": len, "sum": sum, "min": min, "max": max,
        "abs": abs, "round": round, "any": any, "all": all,
        "sorted": sorted, "int": int, "float": float, "str": str, "bool": bool,
    }
    namespace = {"__builtins__": safe_builtins, "df": df, "pd": pd}

    try:
        return True, eval(expression, namespace)
    except Exception as exc:
        # Deliberately broad. A narrow list would let an unanticipated error
        # escape into the loop as a crash, which is the one thing this
        # function exists to prevent.
        return False, {
            "error_type": type(exc).__name__,
            "message": str(exc),
            "blocked": False,
        } 


def expected_shape(question):
    """The answer shape this question implies, or None when it has no opinion.

    None is a real answer and the common one. A rule that always guessed would
    reject correct results on unusual phrasing, and a wrong rejection costs a
    retry and can end in refusing a perfectly good answer.
    """
    text = f" {question.lower()} "

    if any(marker in text for marker in _GROUPED_MARKERS):
        return "series"
    if any(marker in text for marker in _ROWS_MARKERS):
        return "table"
    if any(marker in text for marker in _SCALAR_MARKERS):
        return "scalar"
    return None


def observe(ok, payload, df, question=None):
    """Judge whether a result actually answers anything.

    THIS IS THE PART THAT MAKES THE LOOP A LOOP.
    Retrying on an exception is not observation — it accepts every expression
    that RUNS, including the ones that run fine and answer nothing. An empty
    Series, a whole 15,000-row frame and a None are all successful
    evaluations, and none of them is an answer.

    `question` is optional and powers the last check only. Without it the
    first three still apply.

    Returns
    -------
    dict — usable, reason, retryable, hint, result.

    `retryable` is the loop's stop condition, and the distinction it draws is
    the whole value of the check: a KeyError means the column does not exist,
    so a second attempt produces the same KeyError. An empty result means the
    column exists and the VALUE missed, which a corrected expression can fix.
    """
    # --- 1. did it run at all -------------------------------------------
    if not ok:
        return {
            "usable": False,
            "reason": f"{payload['error_type']}: {payload['message']}",
            # A blocklist hit is a refusal, not a mistake. Feeding it back
            # would invite a second attempt at the same forbidden thing.
            "retryable": (
                not payload["blocked"]
                and payload["error_type"] in RETRYABLE_ERRORS
            ),
            "hint": _hint_for(payload),
            "result": None,
        }

    result = payload

    # --- 2. is the shape usable -----------------------------------------
    # None comes back from print(...) and from in-place methods. The
    # expression ran; it just evaluated to nothing.
    if result is None:
        return {
            "usable": False,
            "reason": "the expression evaluated to None",
            "retryable": True,
            "hint": (
                "an expression must evaluate to the answer itself. print() "
                "and inplace=True both return None."
            ),
            "result": None,
        }

    n_rows = len(result) if hasattr(result, "__len__") else 1

    if n_rows == 0:
        return {
            "usable": False,
            "reason": "the expression returned no rows",
            # The column exists — a filter matched nothing. That is fixable,
            # unlike a missing column.
            "retryable": True,
            "hint": (
                "a filter matched nothing. Check the value against the "
                "allowed values in the schema summary — a case mismatch like "
                "'punjab' for 'Punjab' is the usual cause."
            ),
            "result": None,
        }

    if n_rows > MAX_RESULT_ROWS:
        return {
            "usable": False,
            "reason": f"the expression returned {n_rows:,} rows",
            "retryable": True,
            "hint": (
                f"return at most {MAX_RESULT_ROWS} rows. Aggregate, group, or "
                f"use nlargest — the whole frame is not an answer."
            ),
            "result": None,
        }

    # --- 3. is the value sane -------------------------------------------
    # An all-NaN result means the expression ran and measured nothing: usually
    # an aggregation over a column that is empty after filtering.
    if _all_nan(result):
        return {
            "usable": False,
            "reason": "every value in the result is NaN",
            "retryable": True,
            "hint": (
                "the aggregation had no values to work on. Check that the "
                "filter left rows, and that the column is not entirely null "
                "for those rows."
            ),
            "result": None,
        }

    # Every numeric column in this schema is a count, an amount, a rate or a
    # score — all non-negative. `difference` is the one exception and can
    # legitimately be negative, so this is a WARNING rather than a rejection.
    warning = None
    if _has_negative(result):
        warning = (
            "the result contains a negative value. Most columns here cannot "
            "be negative; `difference` is the exception."
        )

    # --- 4. does the shape match what was asked --------------------------
    # A scalar where a breakdown was asked for is the failure this catches:
    # every check above passes, the number is correct, and it answers a
    # different question. Only fires when the phrasing is unambiguous.
    wanted = expected_shape(question) if question else None

    if wanted is not None:
        got = format_result(result)["shape"]
        if got != wanted:
            return {
                "usable": False,
                "reason": (
                    f"the question asks for a {wanted} but the expression "
                    f"returned a {got}"
                ),
                "retryable": True,
                "hint": {
                    "series": (
                        "the question asks for a breakdown — group by the "
                        "column it names."
                    ),
                    "scalar": (
                        "the question asks for one number — aggregate rather "
                        "than grouping."
                    ),
                    "table": (
                        "the question asks for specific rows — use nlargest, "
                        "nsmallest, or a filter."
                    ),
                }[wanted],
                "result": None,
            }

    return {
        "usable": True,
        "reason": None,
        "retryable": False,
        "hint": warning,
        "result": result,
    }


def _hint_for(payload):
    """What to tell the model about this specific failure.

    Generic enough to stay true, specific enough to be actionable. A hint that
    only says "it failed" wastes the retry.
    """
    if payload["blocked"]:
        return None

    return {
        "KeyError": (
            "that column does not exist. Do not guess a similar name — check "
            "the schema summary, and refuse with CANNOT_ANSWER if it is "
            "genuinely absent."
        ),
        "TypeError": (
            "usually a text column being aggregated. churned_12m holds 'Y' "
            "and 'N' and must be mapped to 1 and 0 before mean()."
        ),
        "AttributeError": (
            "that method does not exist on that object — check whether you "
            "have a DataFrame or a Series."
        ),
        "SyntaxError": (
            "an unbalanced bracket or quote, or more than one statement."
        ),
        "ValueError": (
            "often df[['col']] with two brackets where df['col'] with one was "
            "meant."
        ),
    }.get(payload["error_type"])


def _all_nan(result):
    """True when nothing in the result is a real value."""
    if isinstance(result, (pd.Series, pd.DataFrame)):
        return bool(result.isna().all().all())
    return result != result      # NaN is the only value not equal to itself


def _has_negative(result):
    """True when any numeric value in the result is below zero."""
    if isinstance(result, pd.DataFrame):
        numeric = result.select_dtypes("number")
        return bool((numeric < 0).any().any()) if not numeric.empty else False
    if isinstance(result, pd.Series):
        return bool((result < 0).any()) if pd.api.types.is_numeric_dtype(result) else False
    return isinstance(result, (int, float)) and result < 0


def format_result(result):
    """Turn a pandas result into something a narrator and JSON can both take.

    Three problems to solve, none of them cosmetic.

    NUMPY TYPES DO NOT SERIALISE. `df["x"].median()` returns np.float64, not
    float, and FastAPI cannot encode it — the same class of bug that made
    p_value_valid need bool() in the analytics layer.

    JSON HAS NO NaN. `json.dumps(float("nan"))` emits a bare `NaN` token,
    which is invalid JSON and breaks strict parsers. Nulls become None.

    THE RESULT CAN BE ANY SHAPE. A scalar, a Series with a meaningful index,
    or a table. Rather than flattening all three into one format, the shape is
    NAMED — a narrator reading "series" knows the index carries group labels,
    where a bare list of numbers would leave it guessing. observe() compares
    that name against what the question asked for.

    Returns
    -------
    dict — shape, n_rows, value, and truncated.
    """
    # --- scalar ----------------------------------------------------------
    if not isinstance(result, (pd.Series, pd.DataFrame)):
        return {
            "shape": "scalar",
            "n_rows": 1,
            "value": _to_python(result),
            "truncated": False,
        }

    # --- series ----------------------------------------------------------
    # The index is usually group labels — region names, band labels — so it is
    # kept as dict keys rather than discarded. str() on the key because a
    # Categorical or a Timestamp index will not serialise either.
    if isinstance(result, pd.Series):
        trimmed = result.head(MAX_RESULT_ROWS)
        return {
            "shape": "series",
            "n_rows": int(len(result)),
            "name": str(result.name) if result.name is not None else None,
            "value": {
                str(k): _to_python(v) for k, v in trimmed.items()
            },
            "truncated": len(result) > MAX_RESULT_ROWS,
        }

    # --- table -----------------------------------------------------------
    # Records rather than columns: one dict per row reads as rows, which is
    # what the caller asked for when the answer is a table.
    trimmed = result.head(MAX_RESULT_ROWS)
    return {
        "shape": "table",
        "n_rows": int(len(result)),
        "columns": [str(c) for c in result.columns],
        "value": [
            {str(k): _to_python(v) for k, v in row.items()}
            for row in trimmed.to_dict("records")
        ],
        "truncated": len(result) > MAX_RESULT_ROWS,
    }


def _to_python(value):
    """One pandas or numpy value as a plain Python one.

    pandas returns np.float64, np.int64, pd.Timestamp and pd.NA rather than
    float, int, str and None. None of those survive JSON encoding.
    """
    # pd.isna raises on a list or an array, so it is guarded rather than
    # called blind — this function receives whatever a cell happened to hold.
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    # .item() is the numpy scalar's own conversion: np.float64 -> float,
    # np.int64 -> int, np.bool_ -> bool. Safer than guessing the target type.
    if hasattr(value, "item"):
        value = value.item()

    # Rounded to match the Tier 1 tools. Unrounded, a rate comes back as
    # 0.10385756676557864 and a narrator quotes all sixteen digits.
    if isinstance(value, float):
        return round(value, 4)

    if isinstance(value, (int, str, bool)):
        return value

    # Timestamps, Categoricals, anything else: a string a narrator can read.
    return str(value)


def _load_rules():
    """The static half of the prompt.

    Read per call rather than at import, so editing the prompt does not need a
    restart. It is a few kilobytes off local disk — the model call that
    follows costs orders of magnitude more.
    """
    if not _RULES_PATH.exists():
        raise ToolError(
            f"the Tier 3 prompt is missing at {_RULES_PATH}. Tier 3 cannot "
            f"run without its rules."
        )
    return _RULES_PATH.read_text(encoding="utf-8")


def _clean_expression(text):
    """Strip what a model adds around code despite being told not to.

    Fenced blocks are the common one, and a leading 'python' language tag
    inside them. Cheap to remove, and removing it here beats spending a retry
    on formatting when the expression itself was fine.
    """
    text = text.strip()

    # ```python ... ```  or  ``` ... ```
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    # A single trailing newline is normal; anything multi-line is a script,
    # which the contract forbids — let observe() catch it as a SyntaxError
    # rather than silently taking the first line and answering something else.
    return text


def answer_freeform(question, table="default", model=None):
    """Answer a question the parameterised tools cannot express.

    THE LOOP: build the prompt, generate an expression, run it sandboxed,
    OBSERVE the result, and on a retryable failure feed the expression, the
    reason and a hint back for one more attempt.

    Observation is the part that matters. Retrying on an exception alone would
    accept every expression that RUNS — including the whole frame, an empty
    Series, and a scalar where a breakdown was asked for. All three are
    successful evaluations that answer nothing.

    The stop conditions are as important as the retry. A KeyError means the
    column does not exist, so a second attempt produces the same KeyError. A
    blocked expression is a refusal, not a mistake. Both end the loop
    immediately rather than spending an attempt to fail identically.

    WHERE THE LOGGING HAPPENS. Two of the five control points fire inside this
    loop, and both are recorded here rather than where they fire: run_sandboxed
    receives an expression and has no question to log with, and the iteration
    cap is this loop's own condition. Logging is best-effort — record() catches
    its own failures — because a broken log must not break a working answer.

    Parameters
    ----------
    question
        The user's question, in their own words.
    table
        Which frame to run against — "default", "churn" or "segment". The
        ROUTER chooses this: the grain rules that decide it already live in
        tools.md, and the router is where they are applied.
    model
        An override, for tests. Defaults to the light model.

    Returns
    -------
    dict — answered, expression, attempts, result, and reason when it failed.
    The EXPRESSION is returned deliberately: Tier 3 is the one tier whose
    answer nobody validated in advance, so the user sees the code that
    produced their number.
    """
    df = _frame(table)
    llm = model or get_model(light=True)

    rules = _load_rules()
    summary = schema_summary(df, table)

    # Each attempt's failure is appended here, so the second call sees what
    # the first tried and why it did not work. Passing only "it failed" would
    # invite the same expression back.
    feedback = []
    expression = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        prompt = _build_prompt(rules, summary, question, feedback)
        expression = _clean_expression(llm.invoke(prompt).content)

        # The model's own refusal. Not an error — the honest answer when the
        # columns cannot support the question. NOT logged as a guardrail:
        # nothing was blocked, the model answered correctly.
        if expression.startswith(REFUSAL_PREFIX):
            return {
                "answered": False,
                "expression": None,
                "attempts": attempt,
                "result": None,
                "reason": expression[len(REFUSAL_PREFIX):].strip(),
                "refused_by_model": True,
            }

        ok, payload = run_sandboxed(expression, df)

        # CONTROL POINT 4 — the sandbox. Recorded here because run_sandboxed
        # has no question. The expression goes in the reason: a blocked
        # attempt is worth reading later, and it is the only record of what
        # the model tried to run.
        if not ok and payload["blocked"]:
            record(
                "sandbox",
                question,
                "blocked",
                f"pattern '{payload['pattern']}' in: {expression}",
            )

        verdict = observe(ok, payload, df, question=question)

        if verdict["usable"]:
            return {
                "answered": True,
                "expression": expression,
                "attempts": attempt,
                "result": format_result(verdict["result"]),
                "reason": None,
                # observe() passes warnings through on a usable result —
                # a negative value where the schema says there should be none.
                "warning": verdict["hint"],
                "table": table,
            }

        # Stop conditions. A non-retryable failure will fail identically on a
        # second attempt, so spending one is pure cost.
        if not verdict["retryable"]:
            return {
                "answered": False,
                "expression": expression,
                "attempts": attempt,
                "result": None,
                "reason": verdict["reason"],
                "refused_by_model": False,
            }

        feedback.append({
            "expression": expression,
            "reason": verdict["reason"],
            "hint": verdict["hint"],
        })

    # CONTROL POINT 5 — the iteration cap. Reached only when every attempt was
    # retryable and every one failed, which is the case the cap exists to stop.
    # Logged as its own guardrail rather than folded into the sandbox: an
    # exhausted loop and a blocked expression are different failures and their
    # counts should not be added together.
    record(
        "iteration_cap",
        question,
        "exhausted",
        f"{MAX_ATTEMPTS} attempts, last failure: {feedback[-1]['reason']}",
    )

    # Out of attempts. The last failure is the one worth reporting — it is
    # what the model produced after being told what was wrong with the first.
    return {
        "answered": False,
        "expression": expression,
        "attempts": MAX_ATTEMPTS,
        "result": None,
        "reason": (
            f"gave up after {MAX_ATTEMPTS} attempts. Last failure: "
            f"{feedback[-1]['reason']}"
        ),
        "refused_by_model": False,
    }


def _build_prompt(rules, summary, question, feedback):
    """Rules, then columns, then the question — and past failures last.

    Order is deliberate. The rules are stable and belong at the top where a
    model reads most carefully. The failures go last because they are the most
    recent and most specific instruction, and the thing the second attempt
    most needs to act on.
    """
    parts = [
        rules,
        "\n\n--- SCHEMA SUMMARY ---\n",
        summary,
        "\n\n--- QUESTION ---\n",
        question,
    ]

    if feedback:
        parts.append("\n\n--- YOUR PREVIOUS ATTEMPTS FAILED ---\n")
        for i, past in enumerate(feedback, start=1):
            parts.append(f"\nAttempt {i}:\n  {past['expression']}\n")
            parts.append(f"  Failed because: {past['reason']}\n")
            if past["hint"]:
                parts.append(f"  Fix: {past['hint']}\n")
        parts.append(
            "\nWrite a corrected expression. If the failure means the data "
            "cannot answer the question, reply with CANNOT_ANSWER instead of "
            "guessing again.\n"
        )

    parts.append("\nEXPRESSION:")
    return "".join(parts) 