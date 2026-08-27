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

THE LABEL IS OPTIONAL
---------------------
`defaulted` is the model target, and a loan disbursed last week does not have
one yet. So every step that touches it is guarded: present at training time and
checked, absent at scoring time and skipped. Nothing else in this file is
optional.
"""

import pandas as pd

from src.config import (
    LOANS_RAW,                   # path to the raw csv
    LOANS_CLEAN,                 # path to write the cleaned parquet
    LOAN_PURPOSES,               # allowed purpose categories: the 4 legal values
    TERM_MIN, TERM_MAX,          # legal term_months bounds (1, 12)
    INTEREST_MIN, INTEREST_MAX,  # legal interest_rate_pct sanity band (0, 100)
    AS_OF_DATE,                  # this extract's freeze date, passed explicitly
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

def fix_negative_amounts(df):
    """
    Step 7 — illegal value, fate = correct.
    3 rows have negative amount_pkr. Proven in reconciliation to be pure
    sign flips: recovered = inflow_to_loan_ratio * avg_monthly_inflow_pkr
    matched abs(amount_pkr) exactly. So abs() is a recovery, not a guess.
    """
    df = df.copy()
    n_neg = (df["amount_pkr"] < 0).sum()
    if n_neg:
        print(f"[loans] correcting {n_neg} negative amount_pkr (sign flip)")
        df["amount_pkr"] = df["amount_pkr"].abs()
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
    """Freeze purpose to category and defaulted to int8.

    purpose: values are known-clean (exactly the 4 in LOAN_PURPOSES), so lock
    the type in. NOTE: the notebook line was `loan['purpose'].astype('category')`
    WITHOUT assigning back — that returns a converted Series and discards it,
    a no-op. Here we assign it, so the freeze actually takes effect.

    defaulted: arrives as the TEXT 'True'/'False', never as a real boolean.
    It was previously only becoming numeric via pandas read-time inference,
    which is fragile — the inference changed and a groupby('mean') broke with
    "dtype 'str' does not support operation 'mean'". Cast it explicitly here.
    astype(str) first so this holds whether pandas hands us strings or bools.
    map() sends anything unexpected to NaN silently, so validate_loans asserts
    both isin([0,1]) and notna() — this is the model target, it gets guarded.

    The cast is guarded because a scoring upload has no label at all. Without
    the guard this raises KeyError before the validation gate is ever reached,
    so the user would see a crash rather than a message.
    """ 
    
    df["purpose"] = df["purpose"].astype("category")

    if "defaulted" in df.columns:
        df["defaulted"] = df["defaulted"].astype(str).str.strip().map({"True": 1, "False": 0}).astype("int8")

    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def clean_loans(path=LOANS_RAW):
    """Run the full single-table pipeline and return the cleaned frame.

    Order matters: dedup on raw strings first; convert interest before the
    text-strip loop so the numeric column is skipped; parse dates after the
    strip; add the flag and freeze types last.

    No `as_of` parameter: nothing in the cleaning steps uses the anchor. Only
    `validate_loans` does.
    """
    df = load_raw(path)
    df = drop_exact_duplicates(df)
    df = fix_negative_amounts(df) 
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

# Split by whether the column MUST be present. Same shape as the customers
# CATEGORY_COLUMNS split, and for the same reason: a list that mixes mandatory
# and optional columns cannot be checked in one pass without failing the
# label-less upload it is supposed to allow.
REQUIRED = [
    "loan_id", "customer_id", "disbursed_date", "purpose", "amount_pkr",
    "term_months", "interest_rate_pct", "inflow_to_loan_ratio",
]

OPTIONAL_REQUIRED = ["defaulted"]   # present at training time, absent at scoring


def validate_loans(df, as_of):
    """Assert every invariant of the clean loans table. Raises on the first
    violated belief; returns True if all hold.

    `as_of` is the file's freeze date, passed in rather than read from config:
    it describes THIS extract, and a stale anchor fails the future-date check
    on every row of a later upload.
    """
    # key integrity
    assert df["loan_id"].is_unique, "loan_id not unique"

    # categorical membership
    assert df["purpose"].isin(LOAN_PURPOSES).all(), "unexpected purpose value"

    # amount positive — but exempt the parked negatives, which ride in flagged
    assert (df.loc[~df["amount_suspect"], "amount_pkr"] > 0).all(), \
        "non-flagged amount_pkr <= 0"
    
    assert (df["amount_pkr"] >= 0).all(), "negative amount_pkr found"

    # The model target, checked only when it exists. map() sends anything
    # unexpected to NaN silently, so both asserts matter when it does.
    if "defaulted" in df.columns:
        assert df["defaulted"].isin([0, 1]).all(), "defaulted not 0/1"
        assert df["defaulted"].notna().all(), "defaulted has nulls"

    # numeric ranges — legal bounds, not this sample's min/max
    assert df["term_months"].between(TERM_MIN, TERM_MAX).all(), "term out of range"
    assert df["interest_rate_pct"].between(INTEREST_MIN, INTEREST_MAX).all(), \
        "interest out of range"
    assert (df["inflow_to_loan_ratio"] > 0).all(), "ratio must be positive"

    # no future disbursements
    assert (df["disbursed_date"] <= pd.Timestamp(as_of)).all(), \
        "disbursed_date in the future"

    # completeness — mandatory columns always, optional ones only when present
    checked = REQUIRED + [c for c in OPTIONAL_REQUIRED if c in df.columns]
    assert df[checked].notna().all().all(), "NaN in a required column"

    # NOTE: referential integrity (customer_id in customers) is deliberately
    # NOT here — it needs the merge, so it lives in the reconciliation file.
    return True 

# ---------------------------------------------------------------------------
# Entry point — load -> clean -> validate -> save.
# ---------------------------------------------------------------------------

def main():
    df = clean_loans()
    validate_loans(df, AS_OF_DATE)            # gate: raises if anything is wrong
    LOANS_CLEAN.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(LOANS_CLEAN, index=False)   # category + datetime survive parquet
    print(f"loans_clean saved: {df.shape[0]} rows, {df.shape[1]} cols -> {LOANS_CLEAN}")


if __name__ == "__main__":
    main() 