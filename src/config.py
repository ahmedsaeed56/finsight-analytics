from pathlib import Path

ROOT= Path(__file__).resolve().parents[1]

DATA_RAW= ROOT / "data"/ "raw"
DATA_CLEAN= ROOT / "data"/ "clean"

CUSTOMERS_RAW    = DATA_RAW / "customers_raw.csv"
LOANS_RAW        = DATA_RAW / "loans_raw.csv"
TRANSACTIONS_RAW = DATA_RAW / "transactions_raw.csv"

CUSTOMERS_CLEAN  = DATA_CLEAN / "customers_clean.parquet"


# ---------------------------------------------------------------------------
# Customers cleaning schema
# ---------------------------------------------------------------------------
# All the fixed rules the cleaning uses, in one place — so the schema is
# visible and reusable, and changing a threshold is a one-line edit here.

# Dataset "as-of" date: the moment the data was frozen. Reverse-engineered
# from onboarding_date + stored tenure (not the real calendar today).
AS_OF_DATE = "2025-07-01"

# Detection ceiling for contradictory tenures: as-of + ~3 weeks grace,
# absorbing the 30.44 days-per-month approximation wobble.
TENURE_CEILING = "2025-07-22"

# Average days per month (365.25 / 12) — bridges month-counts to the
# day-based arithmetic that date subtraction produces.
DAYS_PER_MONTH = 30.44

# Age bounds and sentinel.
AGE_MIN = 18
AGE_MAX = 100
AGE_SENTINEL = -999

# Whale threshold: monthly inflow above which a customer is an extreme outlier.
WHALE_THRESHOLD = 200_000

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

# columns frozen as plain (unordered) category.
CATEGORY_COLUMNS = ["region", "city", "segment_true"]

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