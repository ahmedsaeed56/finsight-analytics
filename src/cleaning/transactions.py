"""
src/cleaning/transactions.py
============================

Productionized single-table cleaning pipeline for the transactions dataset.

    transactions_raw.csv  ->  clean, validated DataFrame  ->  transactions_clean.parquet

Mirrors the customers and loans pipelines: small single-purpose functions, a
`clean_transactions` orchestrator, a `validate_transactions` gate, and a guarded
`main()` that runs  load -> clean -> validate -> save.

GRAIN — one row per customer per **month**. `customer_id` repeats legitimately
(a customer has many months), so the row's identity is the *pair*
`(customer_id, month)`, not `customer_id` alone. That composite is this table's
equivalent of loans' `loan_id`.

SCOPE — standalone (single table) only. Referential integrity (every
`customer_id` must exist in `customers_clean`) is deliberately NOT enforced
here; it needs the customers table and lives in the reconciliation stage. Note
that transactions carries NO parked/flagged rows — the negative counts were
resolved outright (see `fix_txn_count`), not deferred.
"""

import pandas as pd

from src.config import (
    TRANSACTIONS_RAW,
    TRANSACTIONS_CLEAN,
    AS_OF_DATE,
    PANEL_MONTHS,
) 

# ---------------------------------------------------------------------------
# Individual cleaning steps
# ---------------------------------------------------------------------------

def load_raw(path=TRANSACTIONS_RAW):
    """Read the raw CSV and clean the header labels."""
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    return df


def drop_exact_duplicates(df):
    """Remove exact whole-row duplicates only.

    Same self-validating pattern as loans: whole-row `drop_duplicates` removes
    only rows identical across every column. If a future file ever holds a
    *conflicting* `(customer_id, month)` pair — same customer-month, different
    counts — it survives here and the composite-key assert in the gate fails
    loudly rather than silently keeping one. The raw file had none, but the
    guard belongs in the pipeline for future runs.
    """
    return df.drop_duplicates(keep="first")


def strip_text_values(df):
    """Trim whitespace inside every string column's values (Step 2).

    Load-bearing, not cosmetic. The raw `month` values carried a stray trailing
    space on the 'YYYY-MM' rows, which made the strict `%Y-%m` parse fail on
    every one of them; stripping first is what lets the parse succeed. It also
    cleans `customer_id`, the join key — a padded id like `'C100000 '` would
    silently fail to match customers at the merge, so this must run.
    """
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()
    return df


def fix_txn_count(df):
    """Correct the sign-flipped transaction counts.

    25 rows had a negative `txn_count` sitting beside a normal positive
    `txn_value_pkr` — the value proves the month's activity was real, so only
    the count's sign was corrupted. `abs()` un-flips those and leaves the
    already-positive counts untouched. Resolved *here*, single-table, not
    deferred: `txn_count` exists only in this table, so no join could ever
    recover it — there is nothing to defer to.
    """
    df["txn_count"] = df["txn_count"].abs()
    return df


def parse_month(df):
    """Parse the mixed-format `month` string into a first-of-month datetime.

    Two formats coexist: 'YYYY-MM' (`%Y-%m`) and 'Mon-YYYY' (`%b-%Y`, e.g.
    'Mar-2025'). Parse once per format with `errors="coerce"`, then
    `combine_first` to stitch them. Result is datetime64 pinned to the 1st of
    the month — the day is a documented placeholder; the real resolution is
    monthly (these are monthly aggregates). MUST run after `strip_text_values`,
    or the `%Y-%m` pass fails on the padded strings.
    """
    dash = pd.to_datetime(df["month"], format="%Y-%m", errors="coerce")
    word = pd.to_datetime(df["month"], format="%b-%Y", errors="coerce")
    df["month"] = dash.combine_first(word)
    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def clean_transactions(path=TRANSACTIONS_RAW):
    """Full single-table pipeline. Order matters: strip before `parse_month`,
    because the whitespace fix is what makes the month parse work."""
    df = load_raw(path)
    df = drop_exact_duplicates(df)
    df = strip_text_values(df)
    df = fix_txn_count(df)
    df = parse_month(df)
    return df


# ---------------------------------------------------------------------------
# Validation gate — Step 13.
# Referential integrity is intentionally excluded (it needs the merge).
# ---------------------------------------------------------------------------

REQUIRED = ["customer_id", "month", "txn_count", "txn_value_pkr"]


def validate_transactions(df, as_of):
    """Assert every invariant of the clean transactions table.

    `as_of` is the file's freeze date — see validate_loans.
    """
    # composite key — one row per customer per month (this table's identity)
    assert df.duplicated(["customer_id", "month"]).sum() == 0, "duplicate (customer_id, month)"

    # join key must be clean — no stray whitespace, or the merge silently mismatches
    assert (df["customer_id"] == df["customer_id"].str.strip()).all(), "customer_id has whitespace"

    # counts and values non-negative (negatives were sign-fixed with abs; zero value allowed)
    assert (df["txn_count"] >= 0).all(), "negative txn_count"
    assert (df["txn_value_pkr"] >= 0).all(), "negative txn_value_pkr"

    # no month in the future
    assert (df["month"] <= pd.Timestamp(as_of)).all(), "month in the future"

    # Panel length. Counted as DISTINCT months rather than (max - min), because
    # subtracting two dates gives days, and because a count also catches a file
    # that spans a year with a month missing from the middle.
    #
    # Checked here rather than in the pre-gate: the gate runs before
    # parse_month, where these values are still a mix of 'YYYY-MM' and
    # 'Mon-YYYY' strings and no span is measurable.
    n_months = df["month"].nunique()
    assert n_months == PANEL_MONTHS, \
        f"panel is {n_months} months, expected {PANEL_MONTHS}"

    # completeness
    assert df[REQUIRED].notna().all().all(), "NaN in a required column"

    # NOTE: referential integrity (customer_id in customers) is deliberately NOT
    # here — it needs the merge, so it lives in the reconciliation file.
    return True 
# ---------------------------------------------------------------------------
# Entry point — load -> clean -> validate -> save.
# ---------------------------------------------------------------------------

def main():
    df = clean_transactions()
    validate_transactions(df, AS_OF_DATE)
    TRANSACTIONS_CLEAN.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(TRANSACTIONS_CLEAN, index=False)
    print(f"transactions_clean saved: {df.shape[0]} rows, {df.shape[1]} cols -> {TRANSACTIONS_CLEAN}") 

if __name__ == "__main__":
    main() 