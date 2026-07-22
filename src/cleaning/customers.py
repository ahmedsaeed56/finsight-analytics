"""
Customers cleaning pipeline — production form of notebooks/01_cleaning.ipynb.

This is the notebook's decided steps, refactored into small composed functions
(one per stage) orchestrated by `clean_customers`. Same logic, same order, same
schema — now callable and rerunnable on any raw customers file.

Explore in the notebook, productionize here.
"""

import numpy as np
import pandas as pd

from src import config as C


# ==========================================================================
# Load-correctness — structural fixes that unblock everything else
# ==========================================================================
def strip_headers(df):
    # column names carry trailing spaces ('age ' -> 'age'); .str.strip cleans
    # them. Without this, every cust["age"]-style lookup fails.
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


def strip_string_values(df):
    # every text (object) column has trailing whitespace ('Sindh   ' -> 'Sindh').
    # Loop the object columns and strip each; assign back into the column.
    df = df.copy()
    for col in df.select_dtypes("object"):
        df[col] = df[col].str.strip()
    return df


# ==========================================================================
# Step 2 — sentinels
# ==========================================================================
def normalize_sentinels(df):
    # age can arrive as text when the raw column mixes numbers with sentinels,
    # so coerce to numeric first (non-numbers -> NaN), then convert the -999
    # sentinel (a code meaning "unknown") to real NaN.
    df = df.copy()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["age"] = df["age"].replace(C.AGE_SENTINEL, np.nan)
    return df


# ==========================================================================
# Step 3 — types
# ==========================================================================
def parse_dates(df):
    # onboarding_date mixes three formats. One format only rescues its own
    # rows (others -> NaT), so parse each format separately then stitch with
    # combine_first: keep a value where present, fill gaps from the next.
    df = df.copy()
    col = df["onboarding_date"].astype(str).str.strip()
    dash = pd.to_datetime(col, format="%Y-%m-%d", errors="coerce")
    slash = pd.to_datetime(col, format="%d/%m/%Y", errors="coerce")
    word = pd.to_datetime(col, format="%b %d, %Y", errors="coerce")
    df["onboarding_date"] = dash.combine_first(slash).combine_first(word)
    return df


def standardize_booleans(df):
    # the six-costume yes/no columns (Yes/True/1/N/False/0) -> clean 1/0.
    # replace with the shared map, then astype(int) so they're truly numeric.
    df = df.copy()
    for col in C.BOOLEAN_COLUMNS:
        df[col] = df[col].replace(C.BOOLEAN_MAP).astype(int)
    return df


def freeze_categories(df):
    # low-cardinality label columns -> category dtype (memory + validity).
    # income_band is ORDERED (rank <25k < ... < 250k+ preserved for modeling).
    # Done after values are cleaned, so only clean labels get frozen.
    df = df.copy()
    for col in C.CATEGORY_COLUMNS:
        df[col] = df[col].astype("category")
    df["declared_income_band"] = pd.Categorical(
        df["declared_income_band"],
        categories=C.BAND_ORDER,
        ordered=True,
    )
    return df


# ==========================================================================
# Step 5 — value standardization
# ==========================================================================
def standardize_values(df):
    # region: fold 16 case-variants -> 6 canonical (.replace leaves unlisted
    # values visible, unlike .map which would NaN them).
    # income-band: fold the 25k-50k variant into 25-50k.
    df = df.copy()
    df["region"] = df["region"].replace(C.REGION_MAP)
    df["declared_income_band"] = df["declared_income_band"].replace(C.INCOME_BAND_MAP)
    return df


# ==========================================================================
# Step 9 — uniqueness (early: must precede any train/test split)
# ==========================================================================
def drop_dupes(df):
    # 200 exact duplicate rows (all confirmed exact copies) -> keep first.
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    assert df["customer_id"].is_unique, "customer_id not unique after dedup"
    return df


# ==========================================================================
# Step 7 — range & cross-field validation
# ==========================================================================
def nullify_impossible_ages(df):
    # any age outside [18,100] is an impossible/typo value (e.g. 250) with no
    # defensible recovery -> nullify (honest "unknown" over fabrication).
    # Sentinels are already NaN; this catches the remaining out-of-range ones.
    df = df.copy()
    bad = ~df["age"].between(C.AGE_MIN, C.AGE_MAX) & df["age"].notna()
    df.loc[bad, "age"] = np.nan
    return df


def recover_tenure(df):
    # tenure is RECOVERABLE from onboarding_date + the as-of anchor, not
    # guessed. Two jobs:
    #   1. Fill missing (NaN) tenure with (anchor - onboarding) in months.
    #   2. Correct CONTRADICTORY stored tenure: rows whose onboarding + stored
    #      tenure overshoots the grace ceiling (impossible future) get
    #      overwritten with the computed value.
    df = df.copy()
    anchor = pd.Timestamp(C.AS_OF_DATE)
    ceiling = pd.Timestamp(C.TENURE_CEILING)

    # correct tenure implied by the reliable dates
    gap_days = (anchor - df["onboarding_date"]).dt.days
    number = (gap_days / C.DAYS_PER_MONTH).round()

    # where does onboarding + STORED tenure land? (detects contradictions)
    as_off = pd.to_timedelta(df["wallet_tenure_months"] * C.DAYS_PER_MONTH, unit="D")
    the_date = df["onboarding_date"] + as_off

    # 1. fill the blanks
    df["wallet_tenure_months"] = df["wallet_tenure_months"].fillna(number)

    # 2. overwrite the contradictory rows only
    contradictory = the_date > ceiling
    df.loc[contradictory, "wallet_tenure_months"] = number[contradictory]

    return df


# ==========================================================================
# Step 10 (flags) — snapshot the NaN pattern BEFORE filling
# ==========================================================================
def add_missing_flags(df):
    # flags record WHERE the NaN are — must run before fillna, or isna() sees
    # nothing. Flags carry no statistic, so they're safe to make here.
    df = df.copy()
    df["age_missing"] = df["age"].isna().astype(int)
    df["income_band_missing"] = df["declared_income_band"].isna().astype(int)
    return df


def impute_missing(df):
    # STATISTICAL fills. NOTE: mode/median are learned from the data, so for a
    # leakage-safe pipeline these should be fit on the TRAIN split only. This
    # all-in-one version mirrors the exploratory notebook; move this call to
    # post-split when wiring up modeling.
    #   age -> median (robust numeric centre, resists outliers)
    #   income_band -> mode()[0] (natural fill for a category; [0] because mode
    #                  returns a Series)
    df = df.copy()
    df["age"] = df["age"].fillna(df["age"].median())
    df["declared_income_band"] = df["declared_income_band"].fillna(
        df["declared_income_band"].mode()[0]
    )
    return df


# ==========================================================================
# Step 11 — derived columns
# ==========================================================================
def add_derived_columns(df):
    # tenure_years: reporting convenience (same info as months, rescaled).
    # is_whale: 1/0 flag marking the extreme-inflow tail (28 whales) so models
    # get an explicit handle on outliers instead of being distorted by them.
    df = df.copy()
    df["tenure_years"] = (df["wallet_tenure_months"] / 12).round(2)
    df["is_whale"] = (df["avg_monthly_inflow_pkr"] > C.WHALE_THRESHOLD).astype(int)
    return df


# ==========================================================================
# Orchestrator
# ==========================================================================
def clean_customers(raw):
    # full pipeline: raw -> clean, in the notebook's decided order.
    df = raw.copy()
    df = strip_headers(df)
    df = strip_string_values(df)
    df = normalize_sentinels(df)        # Step 2
    df = parse_dates(df)                # Step 3
    df = standardize_values(df)         # Step 5
    df = standardize_booleans(df)       # Step 5
    df = drop_dupes(df)                 # Step 9 (before split)
    df = nullify_impossible_ages(df)    # Step 7
    df = recover_tenure(df)             # Step 7
    df = add_missing_flags(df)          # Step 10 flags (pre-fill)
    df = impute_missing(df)             # Step 10 fill (see leakage note)
    df = add_derived_columns(df)        # Step 11
    df = freeze_categories(df)          # type freeze (after values clean)
    return df


# ==========================================================================
# Step 13 — validation gate
# ==========================================================================
def validate(df):
    # each assert encodes a belief the cleaning established; raises if broken.
    assert df["customer_id"].is_unique, "customer_id not unique"
    assert df["age"].notna().all(), "age still has NaN"
    assert df["wallet_tenure_months"].notna().all(), "tenure still has NaN"
    assert df["declared_income_band"].notna().all(), "income-band still has NaN"
    assert df["age"].between(C.AGE_MIN, C.AGE_MAX).all(), "age out of range"
    assert df["wallet_tenure_months"].between(0, 120).all(), "tenure out of range"
    assert df["onboarding_date"].max() <= pd.Timestamp(C.TENURE_CEILING), "future onboarding"
    assert set(df["region"].dropna().unique()) <= set(C.VALID_REGIONS), "invalid region"
    assert set(df["segment_true"].dropna().unique()) <= set(C.VALID_SEGMENTS), "invalid segment"
    for col in ["smartphone_user", "has_savings", "has_insurance", "is_whale"]:
        assert set(df[col].unique()) <= {0, 1}, f"{col} not binary"
    print("All validation checks passed ✅")
    return df


# ==========================================================================
# Entry point — reproduce data/clean/customers_clean.parquet from raw
# ==========================================================================
def main():
    raw = pd.read_csv(C.CUSTOMERS_RAW, skipinitialspace=True)
    cust = clean_customers(raw)
    validate(cust)
    C.DATA_CLEAN.mkdir(parents=True, exist_ok=True)
    cust.to_parquet(C.CUSTOMERS_CLEAN, index=False)
    print("Saved:", C.CUSTOMERS_CLEAN)
    return cust


if __name__ == "__main__":
    main() 