"""
Band definitions — continuous columns cut into named groups for analytics.

WHY THESE EXIST
---------------
The Tier 1 `aggregate_metric` tool can only group by low-cardinality columns.
Grouping by `age` directly would produce ~50 one-row groups; grouping by
`inflow_to_loan_ratio` would produce thousands. But "default rate by loan-to-
income band" is exactly the question a user asks, and it is exactly what the
EDA answered.

So the bands are precomputed here and land in the feature parquets as ordinary
categorical columns.

WHY NOT LET THE LLM CHOOSE THE CUT POINTS
-----------------------------------------
A `derive` parameter taking bins at call time was considered and rejected. The
LLM would supply the numbers, and while it would usually copy them correctly
from the schema file, nothing would catch the occasional wrong cut — the user
would get a real-looking answer built on an arbitrary boundary. Baking them in
means they cannot be wrong.

ANALYTICS ONLY
--------------
These columns are never model features. They are coarser versions of continuous
columns the models already use, so feeding both would add redundancy for no
predictive gain. The model scripts drop them explicitly.

BOUNDARIES
----------
Taken from notebook 03 so the tool and the case study agree. Quantile-based
bands (credit score, tenure, inflow) are cut on the TRAIN split and the same
edges applied to test — the same fit-on-train discipline as the scaler.

`ratio_band` is deliberately NOT a quantile band; see RATIO_BINS below.
"""
from __future__ import annotations

import pandas as pd

# --- explicit boundaries, chosen for meaning rather than distribution --------

# Age: life-stage bands, from the EDA. cut(), not qcut() — the boundaries mean
# something (early career / established / older) rather than just splitting the
# population into equal quarters.
AGE_BINS = [17, 29, 40, 66]
AGE_LABELS = ["18-29", "30-40", "41-66"]

# Loan-to-inflow ratio. Quartiles were tried and rejected: they returned a
# smooth gradient (6.9 / 8.2 / 11.0 / 30.5) when the actual finding is a
# THRESHOLD — default sits flat until roughly 1.2x monthly inflow and then
# climbs sharply. The shape is what justifies a cap rather than linear
# risk-based pricing, so a band scheme that blurs it would let the narrator
# report the wrong policy conclusion.
#
# 1.24 is where the EDA rates depart from flat; 3.5 is the alternative cap from
# the pricing-convention sensitivity. Both come from the A/B work.
#
# LIMITATION: unlike the quantile bands, these edges are fixed by hand, so a
# regenerated dataset would not move them. That is intentional — the threshold
# is a claim about lending, not a property of this sample — but it means the
# boundaries must be revisited if the data generator changes.
RATIO_BINS = [-float("inf"), 1.24, 3.5, float("inf")]
RATIO_LABELS = ["under 1.2x", "1.2-3.5x", "over 3.5x"]

# Counts that are mostly zero. A mean of "1.3 complaints" misleads when most
# customers have none, so these collapse to presence-and-severity.
COMPLAINTS_BINS = [-1, 0, 1, 100]
COMPLAINTS_LABELS = ["0", "1", "2+"]

FAILED_TXN_BINS = [-1, 0, 1, 2, 100]
FAILED_TXN_LABELS = ["0", "1", "2", "3+"]

DEPENDENTS_BINS = [-1, 0, 2, 100]
DEPENDENTS_LABELS = ["0", "1-2", "3+"]

# --- quantile bands: edges learned on train, applied to test -----------------
# Stored here after being computed once on the train split, so both splits get
# identical cut points. Recompute and update these if the data is regenerated.
QUANTILE_BANDS = {
    "credit_score": "credit_score_band",
    "wallet_tenure_months": "tenure_band",
    "avg_monthly_inflow_pkr": "inflow_band",
}

N_QUANTILES = 4


def _quantile_band(
    series: pd.Series,
    edges: list[float] | None,
) -> tuple[pd.Series, list[float]]:
    """Cut into quartiles, returning the band and the edges used.

    On train, `edges` is None and the quartiles are computed. On test, the
    train edges are passed in so both splits share boundaries — a test customer
    in the "high tenure" band means the same thing as a train customer in it.

    Edges are widened to +/- infinity so a test value outside the train range
    still lands in a band rather than becoming NaN.
    """
    if edges is None:
        _, edges = pd.qcut(series, N_QUANTILES, retbins=True, duplicates="drop")
        edges = list(edges)
        edges[0], edges[-1] = float("-inf"), float("inf")

    labels = [f"Q{i + 1}" for i in range(len(edges) - 1)]
    return pd.cut(series, bins=edges, labels=labels), edges


def add_bands(
    df: pd.DataFrame,
    edges: dict[str, list[float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    """Add every band column present in this table.

    Parameters
    ----------
    edges
        Quantile edges from the train split. Pass None when building train
        (edges are computed and returned); pass the returned dict when building
        test. `ratio_band` is not affected — its edges are fixed.

    Returns
    -------
    The frame with band columns added, and the edges used — so the caller can
    hand them to the test build.
    """
    out = df.copy()
    used = dict(edges or {})

    if "age" in out.columns:
        out["age_band"] = pd.cut(out["age"], bins=AGE_BINS, labels=AGE_LABELS)

    if "inflow_to_loan_ratio" in out.columns:
        out["ratio_band"] = pd.cut(
            out["inflow_to_loan_ratio"], bins=RATIO_BINS, labels=RATIO_LABELS
        )

    if "complaints_12m" in out.columns:
        out["complaints_band"] = pd.cut(
            out["complaints_12m"], bins=COMPLAINTS_BINS, labels=COMPLAINTS_LABELS
        )

    if "failed_txns_12m" in out.columns:
        out["failed_txns_band"] = pd.cut(
            out["failed_txns_12m"], bins=FAILED_TXN_BINS, labels=FAILED_TXN_LABELS
        )

    if "dependents" in out.columns:
        out["dependents_band"] = pd.cut(
            out["dependents"], bins=DEPENDENTS_BINS, labels=DEPENDENTS_LABELS
        )

    for col, band_name in QUANTILE_BANDS.items():
        if col in out.columns:
            band, e = _quantile_band(out[col], used.get(col))
            out[band_name] = band
            used[col] = e

    return out, used


# Every band column this module can produce. The model scripts drop these —
# they are analytics-only, and would otherwise reach the ColumnTransformer as
# unhandled categoricals.
BAND_COLUMNS = [
    "age_band",
    "complaints_band",
    "failed_txns_band",
    "dependents_band",
    "ratio_band",
    "credit_score_band",
    "tenure_band",
    "inflow_band",
]  