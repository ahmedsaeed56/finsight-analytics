"""
src/pipeline/drift.py
=====================

Has the population changed since the model was fitted?

    one feature frame + the training baseline  ->  a PSI report per column

INPUT DRIFT, NOT CONCEPT DRIFT
------------------------------
This compares INPUTS against INPUTS: today's credit scores against the ones
the model trained on. That needs no labels and is measurable the day a file
lands.

What it cannot see is CONCEPT drift — the relationship changing rather than
the people. If a credit score of 480 used to mean 14% default risk and now
means 22%, every distribution here looks identical and the model has quietly
stopped working. Catching that needs labels, and labels need months. Input
drift is the early warning; performance decay is the later confirmation.

WHY PSI AND NOT A KS TEST
-------------------------
KS compares cumulative distributions, which needs an ordering — so it cannot
handle `region` at all, and PSI would be required for the categoricals anyway.
KS also returns a p-value, and on twelve thousand rows every p-value is
significant: it detects differences far too small to act on. PSI measures
MAGNITUDE, and its 0.1 / 0.25 thresholds are what a credit risk team already
reads.

PSI SAYS WHICH COLUMN, NEVER WHY
--------------------------------
A high score on `region` is equally consistent with the company expanding into
Balochistan, an export filtering wrongly, and a month-long promotion. Same
number, three meanings, three different responses. So this reports and a human
decides — the same detect-and-report rule that governs new categorical values.

FLAGS, NEVER BLOCKS
-------------------
Nothing here stops a file being scored. The volume-guardrail finding is what
makes that safe: bands vary treatment rather than access, so a drifted model
degrades into badly prioritised manual review, not creditworthy applicants
refused.
"""

import json
import math

from src.config import BASELINE
from src.models.baseline import numeric_shares

# Credit-risk convention. These are the numbers a risk team already reads,
# which is most of why PSI was chosen over a statistical test.
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25

# An empty bin makes ln(0) undefined, and a zero baseline share divides by
# zero. Replacing a zero with a tiny number lets the formula EXPRESS the shift
# — an empty bin where training held 10% is a large move, and this reports it
# as one instead of crashing.
ZERO_FLOOR = 0.0001


def psi(baseline_shares, current_shares):
    """Population Stability Index between two distributions.

    Per bin: (current - baseline) * ln(current / baseline), summed.

    Both arguments are lists of shares in the SAME bin order, each summing to
    roughly 1. Order is what makes the comparison meaningful — bin three
    against bin three — so both sides must come from the same edges. That is
    why numeric_shares is imported from baseline.py rather than reimplemented
    here: two implementations of "what is a bin" is how this reported 1.58 on
    identical data.
    """
    total = 0.0

    for base, current in zip(baseline_shares, current_shares):
        base = max(base, ZERO_FLOOR)
        current = max(current, ZERO_FLOOR)
        total += (current - base) * math.log(current / base)

    return total


def band(score):
    """The conventional reading of a PSI score."""
    if score < PSI_MODERATE:
        return "stable"
    if score < PSI_SIGNIFICANT:
        return "moderate"
    return "significant"


def categorical_shares(series, baseline_shares):
    """Share of rows per level, aligned to the baseline's levels.

    Two things can differ, and both matter. A level the baseline knows may be
    absent here — it must show 0, not vanish, or the two lists stop lining up.
    And a level the baseline has never seen may appear, which is drift of the
    most consequential kind: analytics can group by it, but the model's
    encoder cannot score it.
    """
    shares = series.value_counts(normalize=True)
    shares.index = shares.index.astype(str)

    levels = list(baseline_shares) + [
        level for level in shares.index if level not in baseline_shares
    ]

    base = [baseline_shares.get(level, 0.0) for level in levels]
    current = [float(shares.get(level, 0.0)) for level in levels]
    unseen = [level for level in shares.index if level not in baseline_shares]

    return base, current, unseen


def load_baseline():
    """The training population's statistics, from disk."""
    if not BASELINE.exists():
        raise FileNotFoundError(
            f"no baseline at {BASELINE}. Run `python -m src.models.baseline` "
            f"after training to create it."
        )
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def check_drift(df, table, baseline=None):
    """Compare one uploaded feature table against its training baseline.

    Returns findings, never raises — same contract as the pre-gate and
    reconciliation, and for the same reason: the caller reports everything at
    once and decides what to do.

    A column in the baseline but missing from the upload is reported rather
    than skipped. Its absence is itself a finding.
    """
    baseline = baseline or load_baseline()

    if table not in baseline:
        raise ValueError(
            f"no baseline for table '{table}'. Available: {sorted(baseline)}"
        )

    section = baseline[table]["columns"]
    columns = {}

    for col, stats in section.items():
        if col not in df.columns:
            columns[col] = {
                "psi": None,
                "band": "missing",
                "note": "column is in the baseline but not in this upload",
            }
            continue

        series = df[col].dropna()

        if series.empty:
            columns[col] = {
                "psi": None,
                "band": "missing",
                "note": "every value in this column is null",
            }
            continue

        entry = {}

        if stats["kind"] == "numeric":
            # Both sides through the same function, and the baseline's shares
            # were MEASURED at build time rather than assumed to be equal.
            current = numeric_shares(series, stats["edges"])
            entry["psi"] = round(psi(stats["shares"], current), 4)

            # The per-row range check lives in scoring.py, but the population
            # view belongs here: how much of this file sits outside anything
            # the model was trained on.
            below = int((series < stats["min"]).sum())
            above = int((series > stats["max"]).sum())
            if below or above:
                entry["out_of_range"] = {
                    "below_min": below,
                    "above_max": above,
                    "trained_on": [stats["min"], stats["max"]],
                }
        else:
            base, current, unseen = categorical_shares(series, stats["shares"])
            entry["psi"] = round(psi(base, current), 4)

            if unseen:
                entry["unseen_levels"] = sorted(unseen)

        entry["band"] = band(entry["psi"])
        columns[col] = entry

    flagged = sorted(
        (c for c, e in columns.items()
         if e["band"] in ("moderate", "significant")),
        key=lambda c: columns[c]["psi"],
        reverse=True,
    )

    return {
        "table": table,
        "n_rows": int(len(df)),
        "baseline_rows": baseline[table]["n_rows"],
        # Named and ordered by severity, not a blanket flag — the same reason
        # small_groups names the thin figure rather than caveating the whole
        # answer.
        "flagged": flagged,
        "columns": columns,
        # Never blocks. High drift means look, not stop.
        "ok": not flagged,
    }


def main():
    """Run drift on the reference tables, which must show none.

    This is the regression test for the detector itself: comparing the
    training data against a baseline built from the training data has to give
    PSI near zero on every column. Anything flagged here is a bug in the
    binning, not drift in the data.
    """
    from src.tools.dataset import load_reference, _frame

    load_reference()
    baseline = load_baseline()

    for table in ("default", "churn", "segment"):
        report = check_drift(_frame(table), table, baseline)
        status = "STABLE" if report["ok"] else "FLAGGED"
        print(
            f"{table:<9} {status:<9} {report['n_rows']:>6,} rows "
            f"vs {report['baseline_rows']:,} baseline"
        )
        for col in report["flagged"]:
            entry = report["columns"][col]
            print(f"    {col:<24} PSI {entry['psi']:.4f}  {entry['band']}")


if __name__ == "__main__":
    main() 