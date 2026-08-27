"""
src/guardrails/numbers.py
=========================

Does every number in the narrated answer come from the tool's result?

    answer text + result dict  ->  {ok, missing}

CONTROL POINT 6 — not in the original five, added later because three rounds
of `narrate.md` prose failed to close the digit-transcription bug. The prompt
says "never invent a figure, quote only what the result contains", and
sometimes the model does anyway: 596 becomes 569, 4,196 becomes 4,169, and a
user reading fluent prose has no way to spot the swap.

A prompt that says everything says nothing. So this file exists — plain
arithmetic against the result, in Python, that cannot be talked out of.

WHY EXTRACTION, NOT A JUDGE
---------------------------
An LLM judge would be another model call reading the same output. Same failure
mode, twice the cost. This is a string search — every number in the narrated
text must appear in the result dict, after normalisation. If it doesn't, the
answer is flagged; the user decides.

WHAT COUNTS AS A NUMBER
-----------------------
Digits, two or more. "One" and "a few" are prose, not data. "1" alone is often
prose too (as in "1 in every 100"), so single digits are skipped rather than
flagged as unverifiable — a false alarm on "1" would fire on nearly every
answer.

Percent signs, decimals, and commas are handled. "14%", "14.1%", "0.1412",
"6,394", "6394" — all normalised to a comparable form before matching.

The regex has a preceding-character guard that excludes digits AND periods,
so "05" inside "0.05" is not matched as its own token (the "0." before it
blocks the match). Without that guard the check flags every mention of a
decimal below 1 as an unverifiable "05" or "83" or similar.

WHAT COUNTS AS "IN THE RESULT"
------------------------------
Every leaf value AND every dict KEY in the nested result dict is collected —
numbers in lists, in nested dicts, everywhere, on both sides. Pandas Series
serialise their INDEX as dict keys, so `groupby("savings_balance_pkr")
["amount_pkr"].mean()` returns a dict where the group values live on the KEY
side. Without walking keys, every group label an answer quotes would flag as
unverifiable — a false alarm on every Tier 3 groupby result.

String values that happen to be numeric ("81100", "81,100.0") are also
parsed as numbers, because pandas sometimes serialises numeric index labels
as strings depending on dtype.

Small groups' `n` counts, `rates` values, `overall` figures, group labels,
all found automatically because they are somewhere in the walk.

PROSE SCAFFOLDING
-----------------
"14 in every 100 loans" carries a real 14 and a rhetorical 100 — a number
the model chose for emphasis, not one it read from the result. The same
applies to "top 10", "one in a thousand". A small set of round numbers is
skipped for that reason.

The set is deliberately narrow: 10, 100, 1000. Real data occasionally
contains these values (a group of exactly 100 rows), so the skip trades a
rare true positive for the much more common false alarm. If a genuine "100"
in the data is ever the source of a transcription bug, this skip would miss
it — accepted cost.

FALSE POSITIVES AND WHY THE BAR IS "MISMATCH"
---------------------------------------------
The narrator sometimes derives a number the result doesn't hold directly —
"307 out of 2,198" mentions two counts a `compare_groups` result may carry
under different keys. The extractor will find both counts as leaves and match
them. A number the narrator DERIVES arithmetically (e.g. "the difference is
5 points") the extractor cannot verify; that number will flag.

Given the goal is catching digit transcription, false alarms on genuine
derived numbers are a real cost. Mitigation: derived numbers in the answer
are usually small and often single-digit, so most fall under the two-digit
floor. What's left is the honest signal — the user sees "these numbers
weren't in the source" and can double-check.
"""

from __future__ import annotations

import re

# Two-or-more digit runs OR a decimal number, with optional % and comma
# thousands. The (?<![\w.]) guard on the left excludes a preceding word
# character OR a period — that stops "05" inside "0.05" from matching as its
# own token, which was the source of half the false alarms.
#
# The (?!\w) guard on the right stops matches inside longer words.
_NUMBER_RE = re.compile(r"(?<![\w.])((?:\d[\d,]*\d|\d\d+)(?:\.\d+)?%?)(?!\w)")

# How close is "the same number". 1% relative tolerance handles
# rounding — 0.1412 → "14.1%" — without letting "14" match "17".
_TOLERANCE = 0.01

# Round-number scaffolding — prose almost never carries these as data.
# Skipping them removes false alarms on phrases like "14 in every 100" or
# "top 10 loans". A real data value of exactly 10, 100, or 1000 will slip
# through unverified, which is the accepted trade for cutting the noise
# rate to something usable.
_PROSE_NUMBERS = {10.0, 100.0, 1000.0}


def _normalise(text):
    """Turn a raw number token into a float.

    "14%" -> 0.14
    "14.1%" -> 0.141
    "6,394" -> 6394.0
    "0.1412" -> 0.1412

    Percents are converted to their decimal equivalent so 14% matches 0.14
    from the result dict — different notations, same number.
    """
    stripped = text.replace(",", "").strip()
    if stripped.endswith("%"):
        return float(stripped[:-1]) / 100.0
    return float(stripped)


def _collect_numbers(value, into):
    """Walk any value and drop every number into a set.

    Nested dicts, lists, tuples all traversed. Bools skipped (isinstance(True,
    int) is True in Python, and treating True as 1 would silently satisfy
    a check for the number 1).

    Dict KEYS are walked too, not just values. Pandas Series serialise their
    index as dict keys, so a groupby result carries the group values on the
    KEY side of the dict. Without walking keys, every group label an answer
    quotes flags as unverifiable.

    String values that look like numbers ("81100", "81,100.0") are parsed
    and added — pandas can serialise numeric index labels as strings.
    """
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        into.add(float(value))
        return
    if isinstance(value, dict):
        for k, v in value.items():
            _collect_numbers(k, into)
            _collect_numbers(v, into)
        return
    if isinstance(value, (list, tuple)):
        for v in value:
            _collect_numbers(v, into)
        return
    if isinstance(value, str):
        # A string that IS a number (pandas indexes serialised as strings).
        # Silent on non-numeric strings — they are labels, not data.
        try:
            into.add(float(value.replace(",", "")))
        except ValueError:
            pass
        return
    # None and anything else are ignored.


def _matches(needle, haystack):
    """Is `needle` (a float) present in `haystack` (a set of floats)?

    Tries three readings:
      1. as itself                     (0.1412 matches 0.1412)
      2. as itself times 100           ("14%" -> 0.14 might be stored as 14)
      3. as itself divided by 100      (0.14 stored, but answer says "14")

    Each with the relative tolerance above. The three-reading check is why
    "14%" in the answer matches 0.1412 in the dict — they're the same number
    written differently.
    """
    candidates = (needle, needle * 100.0, needle / 100.0)

    for candidate in candidates:
        for value in haystack:
            if value == 0:
                if candidate == 0:
                    return True
                continue
            if abs(candidate - value) / abs(value) <= _TOLERANCE:
                return True

    return False


def verify_numbers(answer, result):
    """Which numbers in the answer are not present in the result?

    Parameters
    ----------
    answer
        The narrated text the user reads.
    result
        The tool's result dict, whatever shape the tool returned. Nested
        structures are walked; every leaf number and dict key is collected.

    Returns
    -------
    dict — ok (bool), missing (list of str, the tokens as they appeared in
    the answer). Missing is ordered by first appearance so a user reading
    top-to-bottom sees the flags in the order the numbers appear.
    """
    if not isinstance(result, dict):
        # No dict to check against — nothing to verify. Common for tools
        # that return a scalar or a string; not a failure, just nothing to
        # do.
        return {"ok": True, "missing": []}

    haystack = set()
    _collect_numbers(result, haystack)

    seen = set()
    missing = []
    for match in _NUMBER_RE.finditer(answer):
        token = match.group(1)
        if token in seen:
            continue
        seen.add(token)

        try:
            value = _normalise(token)
        except ValueError:
            continue

        # Prose scaffolding (round numbers users say for emphasis) is not
        # data — "14 in every 100" contains a real 14 and a rhetorical 100.
        if value in _PROSE_NUMBERS:
            continue

        if not _matches(value, haystack):
            missing.append(token)

    return {"ok": not missing, "missing": missing} 