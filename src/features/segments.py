"""
Segmentation feature table — one row per customer, all 12 months.

Run as a script to build BOTH splits:
    python -m src.features.segments

NO LEAKAGE RULE
---------------
K-Means predicts nothing, so there is no future to protect and no time window
to enforce. All twelve months are fair game and no customer is dropped.

A different constraint replaces it: segments should describe what kind of
customer someone IS, not what happened to them. So both outcome columns go, and
so do the missingness flags — those describe the data pipeline rather than the
customer, and clustering on them would group people by which fields happened to
be blank.

segment_true is dropped hardest of all. It is the answer key, and it is joined
back only AFTER fitting, for validation.
"""
from __future__ import annotations

import pandas as pd

from src.config import (
    CUSTOMERS_TRAIN, CUSTOMERS_TEST,
    TRANSACTIONS_CLEAN, TRAIN_IDS, TEST_IDS,
    SEGMENT_FEATURES, SEGMENT_FEATURES_TEST,
)
from src.features.bands import add_bands

TXN_COLS = ["total_txns", "total_value", "active_months"]

DROP_COLS = [
    "segment_true",         # answer key — validated against after fitting
    "churned_12m",          # an outcome, not a trait
    "tenure_years",         # duplicate of wallet_tenure_months / 12
    "avg_monthly_txns",     # stored field, unreconcilable with the panel (E4)
    "is_whale",             # 22 customers
    "city",                 # nests in region
    "onboarding_date",      # datetime
    "age_missing",          # pipeline artifact, not a customer trait
    "income_band_missing",  # same
]


def build_segment_features(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    *,
    band_edges: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build the segmentation feature table for one split."""
    txn = transactions.copy()
    txn["is_active"] = txn["txn_count"] > 0

    agg = (
        txn.groupby("customer_id")
        .agg(
            total_txns=("txn_count", "sum"),
            total_value=("txn_value_pkr", "sum"),
            active_months=("is_active", "sum"),
        )
        .reset_index()
    )

    out = customers.merge(agg, on="customer_id", how="left")
    out[TXN_COLS] = out[TXN_COLS].fillna(0)
    out = out.drop(columns=DROP_COLS, errors="ignore")

    # Analytics-only band columns. This table holds all 12,000 customers, so it
    # is the one the Tier 1 tool reads for population-level profile questions.
    out, edges = add_bands(out, band_edges)

    return out.reset_index(drop=True), edges


def _validate(df: pd.DataFrame, split: str) -> None:
    assert "segment_true" not in df.columns, "answer key leaked into features"
    assert "churned_12m" not in df.columns, "outcome column leaked in"
    assert df[TXN_COLS].isna().sum().sum() == 0, "NaN survived the fill"
    assert len(df) > 0, f"{split} table is empty"
    assert df["age_band"].notna().all(), "age fell outside its band edges"
    print(f"  {split}: {df.shape[0]:,} rows x {df.shape[1]} cols")


def main() -> None:
    txns_all = pd.read_parquet(TRANSACTIONS_CLEAN)
    edges = None   # computed on train, reused on test

    for split, cust_path, ids_path, out_path in [
        ("train", CUSTOMERS_TRAIN, TRAIN_IDS, SEGMENT_FEATURES),
        ("test", CUSTOMERS_TEST, TEST_IDS, SEGMENT_FEATURES_TEST),
    ]:
        customers = pd.read_parquet(cust_path)
        ids = pd.read_parquet(ids_path)["customer_id"]
        txns = txns_all[txns_all["customer_id"].isin(ids)].copy()

        table, edges = build_segment_features(customers, txns, band_edges=edges)
        _validate(table, split)
        table.to_parquet(out_path, index=False)

    print("segment features written")


if __name__ == "__main__":
    main()
