"""
Churn feature table — one row per customer, features from months 1-6 only.

Run as a script to build BOTH splits:
    python -m src.features.churn

The same function is called twice, once per split. That is deliberate: a
separate test-side implementation could drift from the train-side one and the
divergence would show up only as unexplained score degradation.

LEAKAGE RULE
------------
churned_12m is a flag over the whole 12-month panel with no date attached.
Features therefore come from months 1-6 (2024-07 .. 2024-12) and the label is
treated as belonging to months 7-12. Using all twelve months would hand the
model the departure itself: a customer who left in month 8 has five dead months
that exist *because* of the outcome.

KNOWN LIMITATION
----------------
Customers who churned inside months 1-6 still carry dead months in their feature
window. A heuristic to infer churn month from trailing inactivity was built and
tested; at its best threshold it was correct for only 64% of flagged customers
and recovered 26% of known churners, so filtering on it would have discarded
more healthy customers than contaminated ones. Rejected on evidence.
"""
from __future__ import annotations

import pandas as pd

from src.config import (
    CUSTOMERS_TRAIN, CUSTOMERS_TEST,
    TRANSACTIONS_CLEAN, TRAIN_IDS, TEST_IDS,
    CHURN_FEATURES, CHURN_FEATURES_TEST,
)
from src.features.bands import add_bands

# Defaults for the reference extract. Exposed as parameters below — see the
# docstring for why they cannot stay hardcoded.
PANEL_START = pd.Timestamp("2024-07-01")
WINDOW_END = pd.Timestamp("2024-12-01")   # last month used as a feature
HALF_SPLIT = pd.Timestamp("2024-10-01")   # first month of the second half

TXN_COLS = ["total_counts", "total_amount", "active_months",
            "first", "last", "difference"]

DROP_COLS = [
    "segment_true",      # K-Means answer key — must never reach a model
    "tenure_years",      # duplicate of wallet_tenure_months / 12
    "avg_monthly_txns",  # stored field, unreconcilable with the panel (E4)
    "is_whale",          # 22 customers
    "city",              # nests in region; extremes sat on thin groups
    "onboarding_date",   # datetime; tenure encodes it numerically
]


def build_churn_features(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    *,
    panel_start: pd.Timestamp = PANEL_START,
    window_end: pd.Timestamp = WINDOW_END,
    half_split: pd.Timestamp = HALF_SPLIT,
    drop_unlabeled: bool = True,
    band_edges: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build the churn feature table for one split.

    Parameters
    ----------
    customers, transactions
        Already filtered to the split being built.
    panel_start, window_end
        First and last month used as features. These describe THIS EXTRACT,
        not the model, so a scorer running against a different panel must pass
        its own.

        The failure mode if they are wrong is the worst kind: `between` matches
        nothing, every customer's aggregates fill to zero, and `_validate`
        still passes because zero is not missing. Not a shifted number — an
        empty one, delivered silently.
    half_split
        First month of the second half. DERIVED, not free: it must be the
        midpoint of [panel_start, window_end] or `difference` compares two
        unequal spans and reads a longer half as growth. The caller computes
        it from the panel rather than being handed an unrelated date.
    drop_unlabeled
        Remove rows with a missing churn label. True for training and
        evaluation. Set False at scoring time — a live customer has no label,
        and dropping on its absence would discard every row you wanted to
        score.
    band_edges
        Quantile cut points from the train split. None when building train
        (they are computed); pass the returned dict when building test, so both
        splits share boundaries.

    Returns
    -------
    The feature table and the band edges used.
    """
    win = transactions[
        transactions["month"].between(panel_start, window_end)
    ].copy()

    # Counted as txn_count > 0, never as row-presence: the dormancy deletion
    # removed dead rows outright, so presence would score a dormant customer
    # as active in every month that survived.
    win["is_active"] = win["txn_count"] > 0

    # True multiplies as 1 and False as 0, so each column carries the real
    # count in its own half of the window and zero in the other. This lets one
    # groupby produce both halves.
    win["first_months"] = win["txn_count"] * (win["month"] < half_split)
    win["last_months"] = win["txn_count"] * (win["month"] >= half_split)

    agg = (
        win.groupby("customer_id")
        .agg(
            total_counts=("txn_count", "sum"),
            total_amount=("txn_value_pkr", "sum"),
            active_months=("is_active", "sum"),
            first=("first_months", "sum"),
            last=("last_months", "sum"),
        )
        .reset_index()
    )

    # Direction, not volume. 10,8,7,4,3,1 and 5,6,5,6,5,6 both total 33 and are
    # identical on every other feature; only this column separates them.
    # A difference is used rather than a ratio because 41 customers have a
    # zero first half, and substituting a value for the resulting infinity
    # would be an arbitrary choice with nothing to justify it.
    agg["difference"] = agg["last"] - agg["first"]

    out = customers.merge(agg, on="customer_id", how="left")

    # A customer absent from the window genuinely transacted zero times.
    # Missing here means "no activity", not "unknown".
    out[TXN_COLS] = out[TXN_COLS].fillna(0)

    if drop_unlabeled:
        out = out.dropna(subset=["churned_12m"])

    out = out.drop(columns=DROP_COLS, errors="ignore")

    # Analytics-only band columns. The models drop these; they exist so the
    # Tier 1 tool can group by them.
    out, edges = add_bands(out, band_edges)

    return out.reset_index(drop=True), edges


def _validate(df: pd.DataFrame, split: str) -> None:
    """Gate every table before it is written.

    Exists because the drop line in the notebook was once commented out while
    the save ran anyway — the parquet came out correct only by accident of
    execution order.
    """
    assert "segment_true" not in df.columns, "answer key leaked into features"
    assert df[TXN_COLS].isna().sum().sum() == 0, "NaN survived the fill"
    assert len(df) > 0, f"{split} table is empty"
    assert df["age_band"].notna().all(), "age fell outside its band edges"
    

    line = f"  {split}: {df.shape[0]:,} rows x {df.shape[1]} cols"

    # Absent at scoring time. Pairs with drop_unlabeled=False in the builder,
    # which already anticipated a label-less run.
    if "churned_12m" in df.columns:
        assert df["churned_12m"].isin(["Y", "N"]).all(), "unexpected label value"
        line += f"  churn rate {(df['churned_12m'] == 'Y').mean():.3f}"

    print(line)


def main() -> None:
    txns = pd.read_parquet(TRANSACTIONS_CLEAN)
    edges = None   # computed on train, reused on test

    for split, cust_path, ids_path, out_path in [
        ("train", CUSTOMERS_TRAIN, TRAIN_IDS, CHURN_FEATURES),
        ("test", CUSTOMERS_TEST, TEST_IDS, CHURN_FEATURES_TEST),
    ]:
        customers = pd.read_parquet(cust_path)
        ids = pd.read_parquet(ids_path)["customer_id"]
        split_txns = txns[txns["customer_id"].isin(ids)].copy()

        table, edges = build_churn_features(
            customers, split_txns, band_edges=edges
        )
        _validate(table, split)
        table.to_parquet(out_path, index=False)

    print("churn features written")


if __name__ == "__main__":
    main() 