"""
Default feature table — one row per loan, bounded by that loan's disbursed_date.

Run as a script to build BOTH splits:
    python -m src.features.default

LEAKAGE RULE
------------
A lender decides on the day of application. So each loan's features are computed
from that customer's transactions strictly BEFORE its own disbursed_date — a
loan approved in September 2024 sees two months of history, one approved in
March 2025 sees eight.

The cutoff varies per row, which no single boolean filter can express. The move
that dissolves it: merge disbursed_date ONTO the transaction rows, so every row
carries its own deadline and the filter becomes a column-vs-column comparison.

KNOWN LIMITATION
----------------
Transactions are monthly; loan dates are daily. A loan disbursed on 20 March
keeps the whole of March, including transactions after the 20th. Cutting at the
start of the disbursement month would remove this but costs every loan its most
recent and most informative month. Kept deliberately, documented here.
"""
from __future__ import annotations

import pandas as pd

from src.config import (
    CUSTOMERS_TRAIN, CUSTOMERS_TEST,
    LOANS_CLEAN, TRANSACTIONS_CLEAN, TRAIN_IDS, TEST_IDS,
    DEFAULT_FEATURES, DEFAULT_FEATURES_TEST,
)
from src.features.bands import add_bands

PANEL_START = pd.Timestamp("2024-07-01")

TXN_COLS = ["total_txns", "total_value", "active_months"]

DROP_COLS = [
    "segment_true",       # K-Means answer key
    "tenure_years",       # duplicate of wallet_tenure_months / 12
    "avg_monthly_txns",   # stored field, unreconcilable with the panel (E4)
    "is_whale",           # 22 customers
    "city",               # nests in region
    "onboarding_date",    # datetime
    "amount_suspect",     # all-False after the upstream fix; zero variance
    "churned_12m",        # the churn model's TARGET — an outcome, not a predictor
    "interest_rate_pct",  # constructed as f(credit_score), corr -0.758; carries
                          # no independent information and splits tree importance
                          # between two copies of one variable
]


def build_default_features(
    customers: pd.DataFrame,
    loans: pd.DataFrame,
    transactions: pd.DataFrame,
    *,
    panel_start: pd.Timestamp = PANEL_START,
    band_edges: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build the default feature table for one split.

    Parameters
    ----------
    panel_start
        Start of the observation panel, used for months_available. Exposed
        because it is a property of this data extract, not of the model — a
        scorer running against a different history depth must pass its own
        value or months_available silently means something else.
    band_edges
        Quantile cut points from the train split. None when building train;
        pass the returned dict when building test.
    """
    cut = transactions.merge(
        loans[["customer_id", "loan_id", "disbursed_date"]],
        on="customer_id",
        how="left",
    )
    # Non-borrowers carry NaT here. Any comparison with NaT is False, so their
    # rows drop out below without needing a special case.
    pre = cut[cut["month"] < cut["disbursed_date"]].copy()
    pre["is_active"] = pre["txn_count"] > 0

    agg = (
        pre.groupby("loan_id")
        .agg(
            total_txns=("txn_count", "sum"),
            total_value=("txn_value_pkr", "sum"),
            active_months=("is_active", "sum"),
        )
        .reset_index()
    )

    out = loans.copy()

    # Computed arithmetically rather than as a groupby size, for two reasons.
    # `size` counts only rows that EXIST, which undercounts because dormancy
    # deleted dead rows outright — that made it a near-duplicate of
    # active_months with almost no independent variance. And loans with no
    # surviving pre-loan history are absent from `agg` entirely, so nothing
    # computed there could reach them.
    #
    # The +1 makes the window inclusive of the disbursement month, matching the
    # date-level row filter above. Without it, active_months exceeds
    # months_available on any loan not disbursed on the 1st.
    out["months_available"] = (
        out["disbursed_date"].dt.to_period("M") - panel_start.to_period("M")
    ).apply(lambda x: x.n) + 1

    out = out.merge(agg, on="loan_id", how="left")
    out[TXN_COLS] = out[TXN_COLS].fillna(0)

    assert (out["active_months"] <= out["months_available"]).all(), \
        "active months exceed the window — check the +1 offset"
    assert out["months_available"].min() >= 1, "zero window would divide by zero"

    # Totals grow with window length: a 10-month loan shows more transactions
    # than a 2-month one for free, with nothing to do with the customer.
    # Dividing removes that. months_available is kept as a feature in its own
    # right so the model knows how much evidence backs each mean — one month is
    # noisier than ten.
    out["average_txns_per_mon"] = out["total_txns"] / out["months_available"]
    out["average_value_per_mon"] = out["total_value"] / out["months_available"]
    out["active_ratio"] = out["active_months"] / out["months_available"]

    out = out.merge(customers, on="customer_id", how="left")
    out = out.drop(columns=DROP_COLS, errors="ignore")

    # Analytics-only band columns. ratio_band matters most here — "default rate
    # by loan-to-income band" is the H1 finding, and it is the question users
    # will ask most often.
    out, edges = add_bands(out, band_edges)

    return out.reset_index(drop=True), edges


def _validate(df: pd.DataFrame, split: str) -> None:
    assert "segment_true" not in df.columns, "answer key leaked into features"
    assert "churned_12m" not in df.columns, "other model's target leaked in"
    assert df[TXN_COLS].isna().sum().sum() == 0, "NaN survived the fill"
    assert len(df) > 0, f"{split} table is empty"
    assert df["ratio_band"].notna().all(), "ratio fell outside its band edges"

    line = f"  {split}: {df.shape[0]:,} rows x {df.shape[1]} cols"

    # Absent at scoring time: a loan disbursed last week has no outcome yet.
    # Guarded rather than removed — when the label IS present it is the model
    # target and still gets checked.
    if "defaulted" in df.columns:
        assert df["defaulted"].notna().all(), "missing target"
        line += f"  default rate {df['defaulted'].astype(int).mean():.3f}"

    print(line) 

def main() -> None:
    loans_all = pd.read_parquet(LOANS_CLEAN)
    txns_all = pd.read_parquet(TRANSACTIONS_CLEAN)
    edges = None   # computed on train, reused on test

    for split, cust_path, ids_path, out_path in [
        ("train", CUSTOMERS_TRAIN, TRAIN_IDS, DEFAULT_FEATURES),
        ("test", CUSTOMERS_TEST, TEST_IDS, DEFAULT_FEATURES_TEST),
    ]:
        customers = pd.read_parquet(cust_path)
        ids = pd.read_parquet(ids_path)["customer_id"]
        loans = loans_all[loans_all["customer_id"].isin(ids)].copy()
        txns = txns_all[txns_all["customer_id"].isin(ids)].copy()

        table, edges = build_default_features(
            customers, loans, txns, band_edges=edges
        )
        _validate(table, split)
        table.to_parquet(out_path, index=False)

    print("default features written")


if __name__ == "__main__":
    main()
