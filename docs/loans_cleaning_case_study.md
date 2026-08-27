# Loans Dataset — Data Cleaning Case Study

## 1. Overview

**Dataset:** `loans_raw.csv` — synthetic JazzCash loan records, built for a product-analytics portfolio project.
**Size:** ~8,050 raw rows → **8,000** after de-duplication; 9 columns.
**Target variable:** `defaulted` (~13.9% positive rate).
**Grain:** one row per **loan**. `customer_id` repeats legitimately (a customer can hold several loans), so the row's identity is `loan_id`, not `customer_id`.

Cleaning followed a fixed **13-step framework** (the same pipeline used on the customers table), applied in a data-driven order and then productionized into `src/cleaning/loans.py`. Every non-trivial decision is recorded below with its reasoning — because the defensibility of a cleaning pipeline lives in *why* each choice was made, not in the code.

**Scope note — standalone only.** Two problems are deliberately *not* solved here because they need the customers table: referential integrity, and recovering three corrupted loan amounts. Both are deferred to a later reconciliation stage. Until then, suspect rows travel into the clean file **flagged**, never silently altered.

## 2. Columns

| column | meaning |
|---|---|
| `loan_id` | unique loan identifier (row key) |
| `customer_id` | borrower; foreign key to the customers table |
| `disbursed_date` | date the loan was paid out |
| `purpose` | loan product type (categorical) |
| `amount_pkr` | loan principal, PKR |
| `term_months` | loan term, months |
| `interest_rate_pct` | annual interest rate, % |
| `inflow_to_loan_ratio` | customer wallet inflow ÷ loan amount |
| `defaulted` | **target** — whether the loan defaulted |

## 3. Step-by-step

### Step 1 — Load & structural fixes
Read with `skipinitialspace=True` (trims the space that follows each delimiter, fixing padded values on read) and stripped the header labels separately, because `skipinitialspace` cleans *values*, not *column names*. Header padding was removed before anything else touched the frame.

### Step 2 — Value normalization
Stripped leading/trailing whitespace *inside* string cell values (e.g. `" merchant_advance "`) across the string columns. The load-time fixes only handled labels and post-delimiter space; padding embedded in the values themselves needed an explicit pass. No coded sentinels (the customers table's `-999`-style missing markers) were present in loans.

### Step 3 — Type conversion
- **`interest_rate_pct`** arrived as text, mixing bare `23.0` with `27.1%`. Stripped `%` and spaces from both ends, then cast with `astype("float")`. `astype` was chosen over `to_numeric(errors="coerce")` on purpose: it **throws** on any non-numeric residue, so a clean run is itself proof nothing survived the strip — a self-validating conversion.
- **`disbursed_date`** was multi-format: ~6,800 ISO `YYYY-MM-DD` rows and ~1,200 `DD/MM/YYYY` slash rows. The slash form is **day-first**, proven by a value like `30/03/2025` — there is no 30th month, so it cannot be month-first. Parsed once per format with `errors="coerce"` (each pass turns the non-matching rows to `NaT`), then merged with `combine_first` so each pass fills the other's gaps. Result: a single `datetime64` column, **0 NaT**. Once parsed, "format" no longer exists — a datetime stores year/month/day as numbers, which is exactly why the three tables' dates never need format reconciliation with each other.
- **`defaulted`** arrived already boolean (`True`/`False`); no work needed.

### Step 4 — Completeness measurement
`isna().sum()` returned **0** across all nine columns. Important caveat: `isna` only detects `NaN`/`NaT`, not disguised missingness (empty strings, sentinel values). Those were ruled out by the value and range checks in Steps 5 and 7 — not assumed clean from this zero.

### Step 5 — Value standardization
`value_counts()` on `purpose` returned four clean values — `nano_loan` (3,650), `merchant_advance` (1,788), `device_finance` (1,406), `emergency` (1,156) — with no casing or typo variants to collapse. The column was frozen to `category`. `defaulted`, already boolean, needed no standardization.

### Step 6 — Membership verification
Asserted `purpose.isin(<allowed four>).all()`, with the allowed set stored as a constant in `config.py`. This is **not** redundant with the `value_counts` above: `value_counts` is a one-time human inspection, whereas the `isin` assertion is machine-checkable and runs on *every* future execution — it fails loudly if a new or misspelt category ever appears, and it lives in the Step 13 gate. `defaulted` needs no such check; a boolean cannot hold anything but its two values.

### Step 7 — Ranges & cross-field (illegal values)
- **`amount_pkr < 0`** — 3 rows. **Decision: parked, not fixed.** Each negative sat beside an otherwise coherent loan (plausible term, interest, ratio), suggesting sign corruption. But the true amount is **recoverable at the merge**: `amount` is a factor inside `inflow_to_loan_ratio`, and the customer's inflow lives in the customers table, so the sign can be confirmed against real data rather than guessed. The rows were flagged (`amount_suspect = True`) and carried into the clean file unresolved. Reasoning: a clean artifact may hold *known-suspect* rows; it may not hold *silently-wrong* ones — and guessing a fix when a real check is one join away would be premature.
- **`term_months`** — range 1–12, no zeros or absurd values. Clean.
- **`interest_rate_pct`** — range 18–36, no negatives or wild rates. Clean.
- **Referential integrity** (every `customer_id` must exist in `customers_clean`) — deferred to the merge, since it needs the customers table loaded. It belongs conceptually in Step 7 as a cross-*table* validity rule; orphan rows would be dropped-and-logged, because a broken foreign key can be neither corrected nor nullified.

### Step 8 — Outlier detection (`inflow_to_loan_ratio`)
This column was the most involved investigation, and its conclusion is the instructive one.

An initial eyeball of ratios above 30 (29 rows, up to 49) showed every one was `merchant_advance` — not scattered noise, a single product. A proper IQR fence was then run: the upper fence (~5.8) flagged 982 rows, all upper-tail, revealing a heavily **right-skewed** column (median 0.82, mean 2.51, max 49). The lower fence came out at **−2.93** — negative — so IQR flagged nothing at the bottom, because a ratio cannot be negative. Inspecting the smallest ratios directly showed the low tail was entirely `nano_loan` (tiny amounts, ratios 0.01–0.03).

**The lesson:** the column mixes loan *products* that live on completely different scales. A ratio of 0.01 is normal for a nano-loan (a small loan to a thin-inflow user); a ratio of 40 is normal for a merchant advance (a small advance against a merchant's large inflow). Pooled together they *had* to look like wild outliers, because a single-population tool (IQR) was measuring several populations stacked in one column. The ratio is only interpretable *within* a product.

**Decision: keep the entire column — no drop, no cap, no winsorize.** Both tails are coherent product segments, not errors; capping would destroy exactly the signal a default model would want. Per-product analysis is an EDA/modeling concern, not a cleaning one.

*(One honest correction during this step: the outlier subset's minimum, 5.80, was briefly mistaken for the column minimum; the true column floor is 0.01. Reading the right table corrected it.)*

### Step 9 — Uniqueness / de-duplication
The raw file held 50 duplicate `loan_id`s, and their pattern was telling: the originals scattered through the file, their copies clustered at the very end (rows ~8001–8048) — the fingerprint of an appended re-ingestion. Every pair was identical field-for-field.

Handled with a whole-row `drop_duplicates(keep="first")` (no subset), followed by `assert loan_id.is_unique`. The two together are **self-validating**: whole-row dedup removes only *exact* copies, so if any duplicate `loan_id` had been *conflicting* (same id, different fields) it would have survived and the uniqueness assert would have thrown. It passed — proving all 50 were exact, with no near-duplicate merge required.

### Step 10 — Missing data
A no-op. `isna` was zero and Steps 5/7 ruled out disguised missingness, so there was nothing to impute — unlike the customers table, which needed imputation for age, income band, and tenure.

### Step 11 — Derived columns
Left intentionally empty. Feature engineering (for instance, bucketing the ratio) is a modeling-stage decision, validated by feature importance — not a cleaning step.

### Step 12 — Uniformity & final formatting
Column names were already `snake_case`; the `amount_suspect` flag was the one new column added. Cosmetic tidy-up only, done last so earlier steps couldn't undo it.

### Step 13 — Validation gate & save
A `validate_loans` assertion battery encodes every belief about the clean data: `loan_id` unique; `purpose` within its allowed set; `amount_pkr > 0` **excluding the flagged rows** (the flag is the carve-out, so the gate does not fail on the parked negatives); `term_months` in 1–12; `interest_rate_pct` within a **legal** band (0–100, deliberately *not* the observed 18–36, so the gate catches a future bad value instead of merely re-confirming this sample); `inflow_to_loan_ratio > 0`; `disbursed_date` on or before the as-of date; and no `NaN` in required columns. Referential integrity is deliberately excluded — it belongs to the merge.

On passing, the frame is written to `loans_clean.parquet`, and the whole pipeline is productionized in `src/cleaning/loans.py` as composed functions + a `clean_loans` orchestrator + the gate + a guarded `main()`, mirroring the customers module, then pushed to GitHub.

## 4. Decisions at a glance
- **Parked the 3 negative amounts** rather than fixing them — recoverable at the merge, so flag now, resolve with real data later.
- **Kept all ratio "outliers"** — they are product segments, not errors; the column mixes populations, so a pooled fence is meaningless.
- **Whole-row dedup + uniqueness assert** — a self-validating pair that proves the duplicates were exact copies.
- **Legal, not observed, bounds in the gate** — so it guards future data, not just this sample.
- **Deferred items flagged in-data** — nothing was silently changed; everything unresolved stays visible and reversible.

## 5. Deferred to reconciliation
- Recover or drop the 3 flagged negative amounts (via `inflow_to_loan_ratio` × customer inflow).
- Referential integrity: confirm every `customer_id` exists in `customers_clean`.
