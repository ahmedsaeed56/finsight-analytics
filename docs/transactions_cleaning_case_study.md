# Transactions Dataset — Data Cleaning Case Study

## 1. Overview

**Dataset:** `transactions_raw.csv` — synthetic JazzCash monthly transaction aggregates, for a product-analytics portfolio project.
**Size:** 178,200 rows, 4 columns.
**Target variable:** none. This is behavioural/activity data — it feeds features (a customer's monthly transaction volume) that join back to the customers and loans tables; it has no label of its own.
**Grain:** one row per **customer per month** (~12 months × ~14,850 customers). `customer_id` repeats legitimately, so the row's identity is the **pair** `(customer_id, month)` — this table's equivalent of loans' single `loan_id`.

Cleaning followed the same fixed 13-step framework as the customers and loans tables, and was productionized into `src/cleaning/transactions.py`. Every non-trivial decision is recorded below with its reasoning.

**Scope note — standalone only.** One check is deliberately deferred: referential integrity (every `customer_id` must exist in `customers_clean`) needs the customers table and belongs to the later reconciliation stage. Unlike loans, this table carries **no parked/flagged rows** — its one data problem (negative counts) was resolved outright, not deferred.

## 2. Columns

| column | meaning |
|---|---|
| `customer_id` | customer; foreign key to the customers table (the join key) |
| `month` | the aggregation month |
| `txn_count` | number of transactions that month |
| `txn_value_pkr` | total transaction value that month, PKR |

## 3. Step-by-step

### Step 1 — Load & structural fixes
Read with `skipinitialspace=True` and stripped the header labels, as in the other two tables.

### Step 2 — Value normalization
Stripped whitespace inside every string column's values. This step was **load-bearing, not cosmetic** — and it was almost missed. The raw `month` values carried a stray trailing space on the `YYYY-MM` rows; that space is invisible in `value_counts` output but breaks a strict date parse (see Step 3). Stripping also cleans `customer_id`, the join key — a padded id like `"C100000 "` would silently fail to match the customers table at the merge. The lesson recorded here: whitespace you cannot see in a display can still break both a parse and a join.

### Step 3 — Type conversion
`month` was a mixed-format string: `YYYY-MM` (`%Y-%m`, e.g. `2024-08`) for nine of the twelve months, and `Mon-YYYY` (`%b-%Y`, e.g. `Mar-2025`) for the other three. Parsed once per format with `errors="coerce"` (each pass `NaT`s the rows it can't match), then merged with `combine_first`. 

A diagnostic moment worth documenting: the first parse left 133,619 `NaT` — exactly the count of the nine `YYYY-MM` months. That number pinpointed the cause: not a missing format, but the `%Y-%m` pass failing on its own rows because of the Step 2 whitespace. After stripping, the same parse produced **0 NaT**. The `NaT`-count check is the safety net that surfaced a hidden defect a visual scan of the values could not.

**Type decision:** the result is a `datetime64` pinned to the **first of each month**. A year-month is arguably a monthly *period* (`period[M]`), but first-of-month datetime was chosen for two practical reasons — it saves cleanly to parquet (period support there is patchy), and it stays uniform with the datetime columns in the other two tables. The day is a documented placeholder; the real resolution is monthly, because these are monthly aggregates.

### Step 4 — Completeness measurement
`isna().sum()` returned 0 across all four columns. As always, this clears only `NaN`; the disguised issue here (whitespace) was caught in Step 2, not by this count.

### Step 5 — Value standardization
No categorical columns to standardize — `customer_id` is an identifier, `month` is now a datetime, and the other two are numeric. A no-op for this table.

### Step 6 — Membership verification
No categorical allowed-set applies. The one membership-style constraint that matters here — every `customer_id` existing in the customers table — is a cross-*table* rule (referential integrity), deferred to the merge.

### Step 7 — Ranges & cross-field (illegal values)
- **`txn_count < 0`** — 25 rows. **Decision: corrected with `abs()`, resolved single-table.** Every negative count sat beside a normal, positive `txn_value_pkr` (e.g. −33 next to 92,460). The value proves the month's activity was real, so only the count's *sign* was corrupted — a sign flip, the same corruption class seen in the loan amounts. Crucially, this could **not** be deferred to the merge the way the loan negatives were: `txn_count` exists only in this table, and no other table stores it or anything it's derivable from, so no join could ever recover the true value. With nothing to defer to, the decision is made here — and `abs()` is the "correct" treatment under the Step 7 hierarchy (un-flip a sign rather than invent a number), applied across all 25.
- **`txn_value_pkr`** — minimum 0.0, no negatives. Zero-value months are unusual but not illegal, so they were noted and left untouched.
- **Future months** — none; asserted in the Step 13 gate (`month <= as-of date`).

### Step 8 — Outlier detection
`txn_count` topped out at 81 and `txn_value_pkr` at ~577,000 — extreme but coherent, consistent with genuinely high-activity customers (merchants), not data errors. Kept as-is. As with the loans ratio, extreme ≠ wrong; these are real behaviour at the top of the distribution.

### Step 9 — Uniqueness / de-duplication
The key shift from loans. `customer_id` repeating is **expected** here — a customer has one row per month — so it is not a de-duplication target. The uniqueness constraint is on the **composite** `(customer_id, month)`: no customer may have two rows for the same month. A `duplicated(["customer_id", "month"])` check returned **empty** — no duplicates, exact or conflicting — so the grain is intact. The production pipeline still applies a defensive whole-row de-dup (mirroring loans, to guard future data), backed by the composite-key assert in the gate.

### Step 10 — Missing data
A no-op — `isna` was zero and the whitespace was resolved in Step 2, leaving nothing to impute.

### Step 11 — Derived columns
Left empty. Any features (e.g. per-customer activity aggregates) belong to the modeling/feature stage, not cleaning.

### Step 12 — Uniformity & final formatting
Column names were already `snake_case`. No new columns were added — unlike loans, this table has no flag column, because its one defect was fixed outright rather than parked.

### Step 13 — Validation gate & save
A `validate_transactions` assertion battery encodes the clean table's invariants: the composite key `(customer_id, month)` is unique; `customer_id` carries no stray whitespace (an explicit guard on the join key, added after the Step 2 lesson); `txn_count >= 0` (the negatives were sign-fixed) and `txn_value_pkr >= 0`; `month` is not in the future; and no `NaN` in required columns. Referential integrity is deliberately excluded — it belongs to the merge.

On passing, the frame is written to `transactions_clean.parquet`, and the pipeline is productionized in `src/cleaning/transactions.py` as composed functions + a `clean_transactions` orchestrator + the gate + a guarded `main()`, mirroring the other two modules.

## 4. Decisions at a glance
- **Sign-fixed the 25 negative counts with `abs`, single-table** — because `txn_count` has no cross-table twin, there is nothing to defer to; contrast the loan amounts, which *were* recoverable at the merge and so were parked.
- **`month` → first-of-month datetime** — parquet-safe and uniform with the other tables; the day is a documented placeholder at monthly resolution.
- **Composite key `(customer_id, month)`** — the grain; `customer_id` repeating is expected, not a duplicate.
- **Whitespace strip was load-bearing** — it unblocked the month parse and protects the join key; the `NaT`-count check is what exposed the hidden space.
- **No parked rows** — the single defect was resolved outright, so nothing is deferred except referential integrity.

## 5. Deferred to reconciliation
- Referential integrity: confirm every `customer_id` exists in `customers_clean`.
