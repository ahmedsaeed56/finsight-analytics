"""
src/pipeline/reconcile.py
=========================

Cross-table checks — the ones each single-table gate deliberately deferred.

    three cleaned frames  ->  one report dict

Both validate_loans and validate_transactions end with the same comment:
referential integrity is NOT here, it needs the merge, it lives in the
reconciliation file. This is that file. It existed only as notebook cells
until now, which meant an upload had no referential check at all.

WHY THAT MATTERS
----------------
A loan whose customer_id is not in customers does not crash anything. It flows
into build_default_features, its how="left" merge finds no transactions, and
total_txns / active_months / active_ratio all fill to zero. The loan is then
scored as a customer with a completely dead wallet. No assertion fires, because
zero is not missing. Silent, and wrong — the same failure shape as the churn
empty-window bug.

DETECTION ONLY
--------------
This module reports. What HAPPENS to an orphan — reject the file, drop those
rows, or keep and flag them — is a policy decision and belongs to the caller.
"""

import numpy as np
import pandas as pd

from src.config import CUSTOMERS_CLEAN, LOANS_CLEAN, TRANSACTIONS_CLEAN

# Tolerance for the ratio reconstruction. The stored ratio is rounded, so exact
# equality would fail on rounding alone; 1% is loose enough to survive that and
# tight enough that a genuinely wrong ratio still fails.
RATIO_TOLERANCE = 0.01


def check_orphans(child, parent, key, label):
    """Rows in `child` whose key does not exist in `parent`.

    One function for both directions — loans->customers and
    transactions->customers — because the logic is identical and two copies
    would drift.

    Returns the count and a sample of the offending keys. Not every key: a
    badly broken file could orphan thousands, and a user needs to see the
    shape of the problem, not all of it.
    """
    unmatched = ~child[key].isin(parent[key])
    orphan_keys = child.loc[unmatched, key].unique()

    return {
        "label": label,
        "n_rows": int(unmatched.sum()),
        "n_keys": len(orphan_keys),
        "sample_keys": sorted(orphan_keys)[:10],
        "ok": not unmatched.any(),
    }


def check_ratio_consistency(loans, customers):
    """Does inflow_to_loan_ratio reconstruct from its two parts?

    ratio should equal amount_pkr / avg_monthly_inflow_pkr. The two live in
    different tables, which is why this cannot sit in validate_loans: a stored
    value is only checkable against the thing it was derived from.

    ORPHANS ARE EXCLUDED. The check runs on a left merge, so an orphan loan
    gets NaN for avg_monthly_inflow_pkr and the comparison yields NaN rather
    than False — reporting a clean result on rows it never actually checked.
    """
    merged = loans.merge(
        customers[["customer_id", "avg_monthly_inflow_pkr"]],
        on="customer_id",
        how="left",
    )

    checkable = merged["avg_monthly_inflow_pkr"].notna()
    subset = merged[checkable]

    reconstructed = subset["amount_pkr"] / subset["avg_monthly_inflow_pkr"]
    agrees = np.isclose(
        reconstructed, subset["inflow_to_loan_ratio"], atol=RATIO_TOLERANCE
    )

    mismatched = subset.loc[~agrees, "loan_id"]

    return {
        "n_checked": int(checkable.sum()),
        "n_skipped_orphans": int((~checkable).sum()),
        "n_mismatched": int((~agrees).sum()),
        "sample_loan_ids": sorted(mismatched)[:10],
        "ok": bool(agrees.all()),
    }


def reconcile(customers, loans, transactions):
    """Run every cross-table check and return one report.

    Takes FRAMES, not paths: by the time this runs the three tables are already
    cleaned and in memory. Reading them from disk again would also mean this
    could only ever check the reference files.

    Same contract as the pre-gate — returns findings, never raises, so the
    caller reports everything at once and decides what to do.
    """
    checks = {
        "loans_to_customers": check_orphans(
            loans, customers, "customer_id", "loans with no matching customer"
        ),
        "transactions_to_customers": check_orphans(
            transactions, customers, "customer_id",
            "transaction rows with no matching customer",
        ),
        "ratio_consistency": check_ratio_consistency(loans, customers),
    }

    return {
        "ok": all(c["ok"] for c in checks.values()),
        "checks": checks,
    }


def main():
    """Run reconciliation on the reference cleaned files, standalone."""
    customers = pd.read_parquet(CUSTOMERS_CLEAN)
    loans = pd.read_parquet(LOANS_CLEAN)
    transactions = pd.read_parquet(TRANSACTIONS_CLEAN)

    report = reconcile(customers, loans, transactions)

    for name, result in report["checks"].items():
        status = "PASS" if result["ok"] else "FAIL"
        print(f"{name:<28} {status}")

    orphan_loans = report["checks"]["loans_to_customers"]
    orphan_txns = report["checks"]["transactions_to_customers"]
    ratio = report["checks"]["ratio_consistency"]

    print(f"\n  orphan loans:        {orphan_loans['n_rows']:,} rows, "
          f"{orphan_loans['n_keys']} customers")
    print(f"  orphan transactions: {orphan_txns['n_rows']:,} rows, "
          f"{orphan_txns['n_keys']} customers")
    print(f"  ratio checked on:    {ratio['n_checked']:,} loans "
          f"({ratio['n_skipped_orphans']:,} skipped as orphans)")
    print(f"  ratio mismatches:    {ratio['n_mismatched']:,}")

    print(f"\nreconciliation: {'PASS' if report['ok'] else 'FAIL'}")
    return report


if __name__ == "__main__":
    main() 