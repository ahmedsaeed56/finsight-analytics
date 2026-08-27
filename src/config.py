from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_CLEAN = ROOT / "data" / "clean"

CUSTOMERS_RAW    = DATA_RAW / "customers_raw.csv"
LOANS_RAW        = DATA_RAW / "loans_raw.csv"
TRANSACTIONS_RAW = DATA_RAW / "transactions_raw.csv"

CUSTOMERS_CLEAN    = DATA_CLEAN / "customers_clean.parquet"
LOANS_CLEAN        = DATA_CLEAN / "loans_clean.parquet"
TRANSACTIONS_CLEAN = DATA_CLEAN / "transactions_clean.parquet"

PROMPTS_DIR = ROOT / "prompts" 

RANDOM_STATE = 42
TEST_SIZE = 0.20

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
DATA_SPLITS = ROOT / "data" / "splits"
DATA_SPLITS.mkdir(parents=True, exist_ok=True)

DATA_FEATURES = ROOT / "data" / "features"
DATA_FEATURES.mkdir(parents=True, exist_ok=True)

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
TRAIN_IDS = DATA_SPLITS / "train_values.parquet"
TEST_IDS  = DATA_SPLITS / "test_values.parquet"

CUSTOMERS_TRAIN = DATA_CLEAN / "customers_train.parquet"
CUSTOMERS_TEST  = DATA_CLEAN / "customers_test.parquet"

# ---------------------------------------------------------------------------
# Feature tables — one pair per model, train and test built by the same function
# ---------------------------------------------------------------------------
CHURN_FEATURES        = DATA_FEATURES / "churn_features.parquet"
CHURN_FEATURES_TEST   = DATA_FEATURES / "churn_features_test.parquet"

DEFAULT_FEATURES      = DATA_FEATURES / "default_features.parquet"
DEFAULT_FEATURES_TEST = DATA_FEATURES / "default_features_test.parquet"

SEGMENT_FEATURES      = DATA_FEATURES / "segment_features.parquet"
SEGMENT_FEATURES_TEST = DATA_FEATURES / "segment_features_test.parquet"

# ---------------------------------------------------------------------------
# Serialised model artifacts
# ---------------------------------------------------------------------------
DEFAULT_MODEL = MODELS_DIR / "default_model.joblib"
CHURN_MODEL   = MODELS_DIR / "churn_model.joblib"
SEGMENT_MODEL = MODELS_DIR / "segment_model.joblib"

# ---------------------------------------------------------------------------
# Imputation values — LEARNED ON TRAIN ONLY, applied unchanged to test.
# ---------------------------------------------------------------------------
AGE_MEDIAN_TRAIN = 33.0
INCOME_BAND_MODE_TRAIN = "25-50k"

# ---------------------------------------------------------------------------
# Risk bands — a policy decision, not a model output
# ---------------------------------------------------------------------------
# The models return a probability. Turning that into an action needs cut points,
# and those are a business choice: a risk officer should be able to move them by
# editing this file, without touching model code.
#
# Both sets were chosen by inspecting the probability distribution on the held-
# out test set and confirming each band separates on the OBSERVED outcome rate.
# The figures below are the measured result at these cuts.
#
# DEFAULT  (test base rate 13.6%)
#   low     <0.30    32.5% of loans    3.3% default (0.24x)   15% of volume
#   medium  0.30-0.60  49.9%          13.8% default (1.01x)   36% of volume
#   high    >0.60      17.6%          32.3% default (2.37x)   49% of volume
#
# The high band carries half the disbursed volume — which is exactly why a
# decline threshold could never clear the 15% volume guardrail. Bands work
# where a threshold did not, because they vary TREATMENT (pricing, limits,
# manual review) rather than access. Nobody is refused.
DEFAULT_BAND_CUTS = (0.30, 0.60)

# CHURN  (test base rate 7.7%)
#   low     <0.40    42.5% of customers   2.2% churn (0.29x)   47% of inflow
#   medium  0.40-0.70  48.5%             10.0% churn (1.30x)   45% of inflow
#   high    >0.70       9.0%             21.1% churn (2.74x)    8% of inflow
#
# Note the inversion against the default model: churn risk concentrates in the
# LOWEST-value customers. The high band holds 24.7% of all churners but under
# 8% of monthly inflow, so a broad retention campaign would spend most of its
# budget defending revenue that was never at risk.
CHURN_BAND_CUTS = (0.40, 0.70)

# ---------------------------------------------------------------------------
# Router confidence gate
# ---------------------------------------------------------------------------
# What the router's confidence score means. A policy choice, like the risk
# band cuts above — moving one is a decision about how often the system asks
# rather than answers.
#
# ROUTER.MD STATES THESE NUMBERS IN PROSE AND MUST BE UPDATED WITH THEM.
# The model calibrates its scores against what the prompt says, so a code
# change alone aims that calibration at the wrong line — and nothing errors,
# the gate simply stops catching what it was meant to catch.
CONFIDENCE_PROCEED = 0.90   # at or above: run without comment
CONFIDENCE_CLARIFY = 0.70   # below: do not run, ask the user instead
                            # between the two: run, and log the choice

BAND_LABELS = ("low", "medium", "high") 

# How many drivers the scoring tools surface per prediction. Three is enough to
# narrate and few enough to stay readable; more reads as a dump.
TOP_DRIVERS_N = 3

# ---------------------------------------------------------------------------
# Pre-gate contract — RAW upload columns, checked before any cleaning runs
# ---------------------------------------------------------------------------
# These are the columns as they appear in the file the user drops in, BEFORE
# anything touches them. Not the cleaned schema: tenure_years, is_whale,
# age_missing, income_band_missing and amount_suspect are all created by
# cleaning and will never be in an upload.
#
# The gate's only job is turning a crash into a message. Without it, a file
# missing loan_id dies twelve steps into cleaning with a KeyError and the
# step-13 gate never runs.
#
# THE GATE MUST STRIP COLUMN NAMES BEFORE COMPARING. All three raw files carry
# TRAILING header padding ('age ', 'region     ', 'purpose         ',
# 'month   '), and skipinitialspace=True does not remove it — it only handles
# spaces after a comma in a VALUE. Unstripped, the gate reports present columns
# as missing.
#
# Extras are REPORTED, never silently ignored: an unrecognised column may be a
# renamed version of one the system needs (loan_income_ratio for
# inflow_to_loan_ratio), and silence would hide that.
#
# LABELS AND THE ANSWER KEY ARE OPTIONAL, deliberately. defaulted and
# churned_12m are outcomes — a loan disbursed last week has no outcome yet, so
# requiring them would reject exactly the scoring upload the system exists to
# serve. segment_true is the K-Means answer key and exists only in the
# synthetic extract. Required for TRAINING, optional for SCORING.

REQUIRED_CUSTOMER_COLS = [
    "customer_id",
    "age",
    "region",
    "city",
    "onboarding_date",
    "wallet_tenure_months",
    "declared_income_band",
    "avg_monthly_inflow_pkr",
    "dependents",
    "smartphone_user",
    "avg_monthly_txns",
    "complaints_12m",
    "failed_txns_12m",
    "has_savings",
    "savings_balance_pkr",
    "has_insurance",
    "credit_score",
]

OPTIONAL_CUSTOMER_COLS = ["segment_true", "churned_12m"]

REQUIRED_LOAN_COLS = [
    "loan_id",
    "customer_id",
    "disbursed_date",
    "purpose",
    "amount_pkr",
    "term_months",
    "interest_rate_pct",
    "inflow_to_loan_ratio",
]

OPTIONAL_LOAN_COLS = ["defaulted"]

REQUIRED_TXN_COLS = [
    "customer_id",
    "month",
    "txn_count",
    "txn_value_pkr",
]

# Every transactions column is load-bearing — nothing optional here.
OPTIONAL_TXN_COLS = []

# ---------------------------------------------------------------------------
# Customers cleaning schema
# ---------------------------------------------------------------------------
# All the fixed rules the cleaning uses, in one place — so the schema is
# visible and reusable, and changing a threshold is a one-line edit here.

# The as-of anchor for THIS extract: the moment the data was frozen. Reverse-
# engineered from onboarding_date + stored tenure (not the real calendar today).
#
# NOT a global fallback. Every cleaning function takes the anchor as an
# argument, and this constant exists so main() can pass it explicitly when
# reproducing the reference build. An upload derives its own anchor from its
# own dates — max(disbursed_date, month) — because the anchor describes the
# FILE, not the world, and a stale one fails every date check in a 2026 file.
AS_OF_DATE = "2025-07-01"

# Grace period for the contradiction ceiling, in days. A tolerance, not a date:
# DAYS_PER_MONTH is an approximation, so onboarding + stored tenure can land
# slightly past the true anchor without actually contradicting it. ~3 weeks
# absorbs that wobble. This does NOT change per upload; the anchor does.
TENURE_GRACE_DAYS = 21

# Average days per month (365.25 / 12) — bridges month-counts to the
# day-based arithmetic that date subtraction produces.
DAYS_PER_MONTH = 30.44

# Age bounds and sentinel.
AGE_MIN = 18
AGE_MAX = 100
AGE_SENTINEL = -999

# Whale threshold: monthly inflow above which a customer is an extreme outlier.
WHALE_THRESHOLD = 200_000

# ---------------------------------------------------------------------------
# Transactions cleaning schema
# ---------------------------------------------------------------------------
# Required panel length. v1 requires a full 12-month upload because the models
# learned on 12-month totals: total_txns = 84 means an active customer only
# because 84 was spread over twelve months. A three-month file showing 21 is
# identical behaviour and reads as a quiet customer. The model has not changed;
# the meaning of the input has.
#
# Arbitrary-window support means normalising every total to a per-month rate,
# which changes what K-Means was fitted on and needs a retrain. That is a real
# v2 feature, not a shortcut taken here.
PANEL_MONTHS = 12  

# region case-variant fold: messy spelling -> canonical. Only dirty variants
# listed; already-correct values pass through .replace untouched.
REGION_MAP = {
    "PUNJAB": "Punjab",
    "punjab": "Punjab",
    "sindh": "Sindh",
    "SINDH": "Sindh",
    "balochistan": "Balochistan",
    "BALOCHISTAN": "Balochistan",
    "kp": "KP",
    "ajk-gb": "AJK-GB",
    "ISLAMABAD": "Islamabad",
    "islamabad": "Islamabad",
}

# income-band value-variant fold.
INCOME_BAND_MAP = {"25k-50k": "25-50k"}

# income band low-to-high — defines the ORDERED category (rank preserved).
BAND_ORDER = ["<25k", "25-50k", "50-100k", "100-250k", "250k+"]

# boolean columns and their six-costume -> 1/0 mapping.
BOOLEAN_COLUMNS = ["smartphone_user", "has_savings", "has_insurance"]
BOOLEAN_MAP = {
    "False": 0,
    "N": 0,
    "1": 1,
    "Yes": 1,
    "True": 1,
    "0": 0,
}

# Columns frozen as plain (unordered) category, split by whether the column
# MUST be present.
#
# region and city describe every customer, so their absence means a broken
# file and freeze_categories should raise on it.
#
# segment_true is the K-Means answer key. It exists only in this synthetic
# extract and will never appear in a real company's upload, so it is guarded
# at freeze time and in validate(), and it stays OUT of the pre-gate's
# required-column list above.
CATEGORY_COLUMNS = ["region", "city"]
OPTIONAL_CATEGORY_COLUMNS = ["segment_true"] 

# allowed sets — used as membership gates and in the validation gate.
VALID_REGIONS = ["Punjab", "Sindh", "KP", "Balochistan", "AJK-GB", "Islamabad"]
VALID_SEGMENTS = ["payroll", "merchant", "borrower", "saver"]

# city -> region lookup, for the cross-field consistency check.
CITY_TO_REGION = {
    "Karachi": "Sindh",
    "Hyderabad": "Sindh",
    "Sukkur": "Sindh",
    "Lahore": "Punjab",
    "Faisalabad": "Punjab",
    "Rawalpindi": "Punjab",
    "Multan": "Punjab",
    "Sargodha": "Punjab",
    "Peshawar": "KP",
    "Abbottabad": "KP",
    "Mardan": "KP",
    "Quetta": "Balochistan",
    "Gwadar": "Balochistan",
    "Islamabad": "Islamabad",
    "Gilgit": "AJK-GB",
    "Muzaffarabad": "AJK-GB",
}

# ---------------------------------------------------------------------------
# Loans cleaning schema
# ---------------------------------------------------------------------------
LOAN_PURPOSES = ["nano_loan", "merchant_advance", "device_finance", "emergency"]
TERM_MIN, TERM_MAX = 1, 12           # legal offered terms — not this sample's range
INTEREST_MIN, INTEREST_MAX = 0, 100  # legal sanity band — deliberately NOT the observed 18–36

# ---------------------------------------------------------------------------
# Human-readable feature names — for narration only, never for modelling.
# ---------------------------------------------------------------------------
# The LLM layer surfaces driver names to end users. "inflow_to_loan_ratio" is a
# column name; "loan-to-income ratio" is what a person reads.
FEATURE_LABELS = {
    "inflow_to_loan_ratio":   "loan-to-income ratio",
    "amount_pkr":             "loan amount",
    "term_months":            "loan term",
    "credit_score":           "credit score",
    "wallet_tenure_months":   "account age",
    "avg_monthly_inflow_pkr": "monthly inflow",
    "complaints_12m":         "complaints in the last year",
    "failed_txns_12m":        "failed transactions",
    "has_savings":            "holds a savings product",
    "has_insurance":          "holds an insurance product",
    "savings_balance_pkr":    "savings balance",
    "dependents":             "dependents",
    "smartphone_user":        "smartphone user",
    "age":                    "age",
    "declared_income_band":   "declared income band",
    "months_available":       "months of wallet history",
    "average_txns_per_mon":   "average monthly transactions",
    "active_ratio":           "share of months active",
    "total_counts":           "transactions in the window",
    "active_months":          "months active",
    "difference":             "change in activity across the window",
    "region":                 "region",
    "purpose":                "loan purpose",
} 

# ---------------------------------------------------------------------------
# Drift baseline — which columns are monitored
# ---------------------------------------------------------------------------
# An ALLOWLIST, not an exclusion list. A column added to the feature build
# later stays out of the baseline until someone puts it here deliberately —
# silent inclusion would mean reporting drift on something nobody chose to
# monitor.
#
# The model features come from the saved artifacts, so they cannot drift out
# of sync with the models. These are the additions: customer attributes worth
# watching even where no model consumes them, because a shift in them still
# means the company's customer base changed.
#
# Excluded on purpose: loan_id and customer_id (identifiers — every value is
# unique, so a distribution shift is meaningless); disbursed_date (a date,
# which moves by definition on every upload); defaulted and churned_12m
# (labels, not inputs — and a scoring upload will not have them); age_missing
# and income_band_missing (pipeline artifacts, not customer attributes); and
# every *_band column (manufactured from columns already listed, so including
# them counts the same drift twice).
DRIFT_PROFILE_COLS = [
    "age",
    "region",
    "wallet_tenure_months",
    "declared_income_band",
    "avg_monthly_inflow_pkr",
    "dependents",
    "smartphone_user",
    "complaints_12m",
    "failed_txns_12m",
    "has_savings",
    "savings_balance_pkr",
    "has_insurance",
    "credit_score",
] 
# The training population's statistics: drift baselines and the quantile band
# edges. Lives beside the models because it is only valid for them — retrain
# and the distributions change too, so pairing a new model with an old
# baseline should not be possible by accident.
#
# JSON rather than joblib: it holds only numbers and strings, and an artifact
# you want to inspect and diff should be readable without Python.
BASELINE = MODELS_DIR / "baseline.json" 