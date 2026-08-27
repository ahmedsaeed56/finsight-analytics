"""
src/models/baseline.py
======================

What the training population looked like, frozen to disk.

    three training feature tables  ->  models/baseline.json

Everything here is a LEARNED PARAMETER: a number computed by measuring the
training data, which something later needs the same value of. That is the same
class of thing as a model coefficient, and it belongs in the same folder for
the same reason — it is only valid for the models it was built alongside.

WHAT IT FEEDS
-------------
Drift detection (src/pipeline/drift.py) compares each upload's distributions
against these, per column. PSI needs the bins to come from the BASELINE, or it
compares two different binnings and the number means nothing.

The per-row range check (src/models/scoring.py) uses min and max to flag a
prediction made outside anything the model was trained on — a credit score of
780 when training ran 324 to 632 gets a number back, and nothing else in the
system would say it is an extrapolation.

Band edges (src/features/bands.py) were always meant to be persisted and never
were. add_bands computes quartile cut points and returns them; every caller so
far has let them die when the process exited, which is why Q1 covers different
credit scores in January than in February.

RUN ONCE, AFTER TRAINING
------------------------
    python -m src.models.baseline
"""

import json

import joblib
import pandas as pd

from src.config import (
    BASELINE,
    CHURN_FEATURES, CHURN_MODEL,
    DEFAULT_FEATURES, DEFAULT_MODEL,
    DRIFT_PROFILE_COLS,
    SEGMENT_FEATURES, SEGMENT_MODEL,
)
from src.features.bands import N_QUANTILES, QUANTILE_BANDS

# Ten is the credit-industry convention for PSI. Columns with heavy ties
# cannot produce ten distinct cut points — active_months holds 0 to 11 with
# thousands of rows at each value — so duplicates are dropped and those
# columns simply get fewer bins.
PSI_BINS = 10

# One entry per model: the table it was fitted on, and the artifact carrying
# its feature list. Read from the artifact rather than hardcoded, so the
# baseline cannot drift out of sync with the model it describes.
_TABLES = [
    ("default", DEFAULT_FEATURES, DEFAULT_MODEL),
    ("churn", CHURN_FEATURES, CHURN_MODEL),
    ("segment", SEGMENT_FEATURES, SEGMENT_MODEL),
]

def numeric_shares(series, edges):
    """Share of rows per bin, cut on a fixed set of edges.

    Lives here and is imported by drift.py so BOTH sides of the comparison bin
    identically. Deriving the baseline's shares instead — on the assumption
    that qcut makes them equal — produced PSI of 1.58 on identical data,
    because duplicates="drop" leaves some columns with two bins holding 70/30
    rather than ten holding 10% each.

    The outer edges are widened to infinity here rather than in the artifact,
    since infinity is not valid JSON. Without it, any value outside the
    training range falls out of every bin and is silently dropped — exactly
    the drift the check exists to catch.
    """
    open_edges = list(edges)
    open_edges[0], open_edges[-1] = float("-inf"), float("inf")

    binned = pd.cut(series, bins=open_edges, include_lowest=True)

    # reindex on the categories, not on what appeared: a bin with no rows must
    # show 0 rather than go missing, or the list shortens and stops lining up
    # against the other side's.
    shares = binned.value_counts(normalize=True, sort=False)
    return shares.reindex(binned.cat.categories, fill_value=0.0).tolist()



def column_stats(df, columns):
    """Per-column statistics for one table.

    Loops the ALLOWLIST, not the frame. A column added to the feature build
    later stays out of the baseline until someone lists it deliberately — the
    same allowlist-over-droplist lesson that made CLUSTER_FEATURES immune when
    eight band columns appeared, while the two droplist models had to be
    re-verified.

    Numeric columns get bin edges (for PSI) plus min and max (for the range
    check). Categoricals get level shares, which serve both.

    Every value is converted to plain Python. numpy's int64 and float64 look
    like numbers and do not survive JSON encoding — the same problem that made
    p_value_valid need bool() and format_result need _to_python.
    """
    stats = {}

    for col in columns:
        if col not in df.columns:
            raise ValueError(
                f"'{col}' is in the baseline column list but not in this "
                f"table. The list and the feature build have diverged."
            )

        value = df[col]

        if pd.api.types.is_numeric_dtype(value):
            edges = pd.qcut(
                value, PSI_BINS, retbins=True, duplicates="drop"
            )[1]
            edges = [float(e) for e in edges]
            stats[col] = {
                "kind": "numeric",
                "min": float(value.min()),
                "max": float(value.max()),
                "edges": edges,
                # Measured, never derived. qcut aims for equal-count bins but
                # duplicates="drop" collapses columns with heavy ties, and the
                # survivors are not equal.
                "shares": numeric_shares(value, edges),
            }
        else:
            shares = value.value_counts(normalize=True)
            stats[col] = {
                "kind": "categorical",
                "shares": {str(k): float(v) for k, v in shares.items()},
            }

    return stats


def band_edges(df):
    """The quartile cut points add_bands would compute for this table.

    MOMENT A — the half of the saved-band-edges problem that never existed.
    Moment B is already wired: all three feature builders accept band_edges
    and forward it to add_bands. What was missing was anything persisting the
    dict, so every upload recomputed its own quartiles and Q1 meant a
    different score range each month.

    Recomputed from the built training parquets rather than by re-running the
    feature build. Same numbers, because those parquets ARE the training
    population.

    THE OUTER EDGES ARE THE REAL MIN AND MAX, NOT INFINITY.
    _quantile_band widens them to +/- inf so a value outside the training
    range still lands in a band rather than becoming NaN, and that widening is
    still required — but it belongs on USE, not in the artifact. Infinity is
    not valid JSON: Python writes it and reads it back happily, while a
    stricter parser, a browser, or another language chokes. Same family as the
    NaN problem in format_result.

    So this file records what was MEASURED, and whoever loads these edges
    widens the ends before passing them to pd.cut. _quantile_band already
    does exactly that.

    NOTE: the three tables produce three different edge sets, because they are
    three different populations — 6,394 loans, 11,760 labelled customers,
    12,000 customers. Stored per table, which preserves current behaviour. The
    open question is whether they SHOULD differ.
    """
    edges = {}

    for col in QUANTILE_BANDS:
        if col not in df.columns:
            continue

        _, cuts = pd.qcut(
            df[col], N_QUANTILES, retbins=True, duplicates="drop"
        )
        edges[col] = [float(c) for c in cuts]

    return edges


def build_baseline():
    """Measure all three training tables and return one artifact.

    Returns a nested dict — table, then column, then the statistics. Nested
    rather than six flat variables, so drift.py reads
    baseline["default"]["credit_score"] and never counts positions.
    """
    baseline = {}

    for name, features_path, model_path in _TABLES:
        df = pd.read_parquet(features_path)

        # Union, not either alone. The model features are what actually
        # affects a score; the profile columns catch a shift in the customer
        # base even where no model consumes the column.
        model_features = joblib.load(model_path)["features"]
        columns = sorted(set(model_features) | set(DRIFT_PROFILE_COLS))

        baseline[name] = {
            "n_rows": int(len(df)),
            "columns": column_stats(df, columns),
            "band_edges": band_edges(df),
        }

    return baseline


def main():
    baseline = build_baseline()

    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text(json.dumps(baseline, indent=2), encoding="utf-8")

    print(f"baseline written -> {BASELINE}")
    for name, section in baseline.items():
        print(
            f"  {name:<9} {section['n_rows']:>6,} rows  "
            f"{len(section['columns'])} columns  "
            f"{len(section['band_edges'])} band edges"
        )

    return baseline


if __name__ == "__main__":
    main() 