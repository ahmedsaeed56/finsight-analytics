"""
src/cleaning/loans.py
=====================

Productionized single-table cleaning pipeline for the loans dataset.

    loans_raw.csv  ->  clean, validated DataFrame  ->  loans_clean.parquet

Mirrors the customers pipeline: small single-purpose functions, a
`clean_loans` orchestrator that composes them, a `validate_loans` gate that
asserts everything we believe about the clean data, and a guarded `main()`
that runs  load -> clean -> validate -> save.

SCOPE — standalone (single table) only.
Two checks are DELIBERATELY not here, because they need the customers table:
    * referential integrity  (every customer_id must exist in customers_clean)
    * resolving the 3 negative amount_pkr rows (recover the sign, or drop)
Both are handled later in the reconciliation stage. Until then the 3
negatives ride into the clean file *flagged* (see `add_flags`) — never
silently altered.
"""

import pandas as pd

from src.config import (
    LOANS_RAW,                   # path to the raw csv
    LOANS_CLEAN,                 # path to write the cleaned parquet  (ADD THIS TO config.py)
    LOAN_PURPOSES,               # allowed purpose categories: the 4 legal values
    TERM_MIN, TERM_MAX,          # legal term_months bounds (1, 12)
    INTEREST_MIN, INTEREST_MAX,  # legal interest_rate_pct sanity band (0, 100)
    AS_OF_DATE,                  # snapshot date; nothing may be disbursed after it
)


# ---------------------------------------------------------------------------
# Individual cleaning steps — one transformation each, testable in isolation
# ---------------------------------------------------------------------------

def load_raw(path=LOANS_RAW):
    """Read the raw CSV and clean the header labels.

    `skipinitialspace=True` trims the space that follows each comma on read,
    fixing padded *values*. It does NOT touch the column *names*, so we strip
    those separately.
    """
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    return df


def drop_exact_duplicates(df):
    """Remove exact whole-row duplicates only.

    `drop_duplicates` with NO subset removes rows identical across *every*
    column. The raw file had 50 duplicate loan_ids and all were exact
    whole-row copies (an appended re-ingestion), so this is safe.

    We intentionally do NOT dedup on loan_id alone. If a future file ever
    holds *conflicting* loan_id duplicates (same id, different fields), they
    survive this step and the `loan_id.is_unique` assert in the gate fails
    loudly — which is what we want, not a silent keep-first of two disagreeing
    rows.
    """
    return df.drop_duplicates(keep="first")


def clean_interest_rate(df):
    """Convert interest_rate_pct from text to float.

    Values arrive mixed: bare '23.0' next to '27.1%'. `.str.strip('% ')`
    removes any '%' or space characters from both ends; `.astype('float')`
    then casts. astype THROWS on anything non-numeric that survives the strip,
    so a clean run is itself proof no residue was left behind.
    """
    df["interest_rate_pct"] = df["interest_rate_pct"].str.strip("% ").astype("float")
    return df


def strip_text_values(df):
    """Trim whitespace INSIDE string cell values.

    The header strip and skipinitialspace only handled labels and
    post-delimiter spaces; this catches padding sitting inside the values
    themselves (e.g. ' merchant_advance '). Runs after interest is converted
    to float, so that column is no longer text and is skipped.

    Note: `select_dtypes('object')` currently also picks up pandas-3 'str'
    columns via a deprecated compatibility bridge — it works but emits a
    warning. Switch to `include=['object', 'str']` when you want to silence it.
    """
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()
    return df


def parse_dates(df):
    """Parse the mixed-format disbursed_date into a single datetime column.

    Two formats coexist: ISO 'YYYY-MM-DD' (dash, ~6800 rows) and 'DD/MM/YYYY'
    (slash, ~1200 rows, day-first — 30/03/2025 proves the slash form can't be
    month-first). Parse the column once per format with errors='coerce' so the
    rows that don't match a pass become NaT, then `combine_first` stitches the
    two Series together, each filling the other's gaps. The result is
    datetime64 — once parsed there is no 'format' left to reconcile.
    """
    slash = pd.to_datetime(df["disbursed_date"], format="%d/%m/%Y", errors="coerce")
    dash  = pd.to_datetime(df["disbursed_date"], format="%Y-%m-%d", errors="coerce")
    df["disbursed_date"] = slash.combine_first(dash)
    return df


def add_flags(df):
    """Flag the negative amount_pkr rows instead of fixing them here.

    The 3 negatives are a sign corruption whose true value we can only confirm
    against the customer's inflow (available at the merge). So we mark them
    with a boolean column and let them travel into the clean file
    KNOWN-suspect. A clean artifact may carry flagged-unresolved rows; it may
    not carry silently-wrong ones. Resolution happens in reconciliation.
    """
    df["amount_suspect"] = df["amount_pkr"] < 0
    return df


def freeze_types(df):
    """Freeze purpose to a category dtype.

    Its values are known-clean (exactly the 4 in LOAN_PURPOSES), so lock the
    type in. NOTE: the notebook line was `loan['purpose'].astype('category')`
    WITHOUT assigning back — that returns a converted Series and discards it,
    a no-op. Here we assign it, so the freeze actually takes effect.
    """
    df["purpose"] = df["purpose"].astype("category")
    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def clean_loans(path=LOANS_RAW):
    """Run the full single-table pipeline and return the cleaned frame.

    Order matters: dedup on raw strings first; convert interest before the
    text-strip loop so the numeric column is skipped; parse dates after the
    strip; add the flag and freeze types last.
    """
    df = load_raw(path)
    df = drop_exact_duplicates(df)
    df = clean_interest_rate(df)
    df = strip_text_values(df)
    df = parse_dates(df)
    df = add_flags(df)
    df = freeze_types(df)
    return df


# ---------------------------------------------------------------------------
# Validation gate — Step 13. Encodes everything we now believe about the data.
# Referential integrity is intentionally excluded (it needs the merge).
# ---------------------------------------------------------------------------

REQUIRED = [
    "loan_id", "customer_id", "disbursed_date", "purpose", "amount_pkr",
    "term_months", "interest_rate_pct", "inflow_to_loan_ratio", "defaulted",
]


def validate_loans(df):
    """Assert every invariant of the clean loans table. Raises on the first
    violated belief; returns True if all hold."""
    # key integrity
    assert df["loan_id"].is_unique, "loan_id not unique"

    # categorical membership
    assert df["purpose"].isin(LOAN_PURPOSES).all(), "unexpected purpose value"

    # amount positive — but exempt the parked negatives, which ride in flagged
    assert (df.loc[~df["amount_suspect"], "amount_pkr"] > 0).all(), \
        "non-flagged amount_pkr <= 0"

    # numeric ranges — legal bounds, not this sample's min/max
    assert df["term_months"].between(TERM_MIN, TERM_MAX).all(), "term out of range"
    assert df["interest_rate_pct"].between(INTEREST_MIN, INTEREST_MAX).all(), \
        "interest out of range"
    assert (df["inflow_to_loan_ratio"] > 0).all(), "ratio must be positive"

    # no future disbursements
    assert (df["disbursed_date"] <= AS_OF_DATE).all(), "disbursed_date in the future"

    # completeness
    assert df[REQUIRED].notna().all().all(), "NaN in a required column"

    # NOTE: referential integrity (customer_id in customers) is deliberately
    # NOT here — it needs the merge, so it lives in the reconciliation file.
    return True


# ---------------------------------------------------------------------------
# Entry point — load -> clean -> validate -> save.
# ---------------------------------------------------------------------------

def main():
    df = clean_loans()
    validate_loans(df)                        # gate: raises if anything is wrong
    LOANS_CLEAN.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(LOANS_CLEAN, index=False)   # category + datetime survive parquet
    print(f"loans_clean saved: {df.shape[0]} rows, {df.shape[1]} cols -> {LOANS_CLEAN}")


if __name__ == "__main__":
    main() 