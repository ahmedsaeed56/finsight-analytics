# JazzCash Fintech — Data Cleaning & Analytics Pipeline

A product-analytics portfolio project built on a synthetic three-table JazzCash
fintech dataset. The focus is a **rigorous, reproducible cleaning pipeline** —
every dataset is cleaned through a fixed 13-step framework, validated by an
assertion gate, and saved as an analysis-ready parquet. The three clean tables
are then reconciled and merged for analysis.

The emphasis throughout is on **defensible decisions**: each non-trivial cleaning
choice is documented with its reasoning, because a cleaning pipeline is only as
trustworthy as the judgment behind it.

## The data

Three tables, linked by `customer_id`:

| table | grain | rows | notes |
|---|---|---|---|
| **customers** | one row per customer | 15,000 | profile, KYC, wallet, segment; the reference table |
| **loans** | one row per loan | 8,000 | `customer_id` repeats; target = `defaulted` (~14%) |
| **transactions** | one row per customer per month | 178,200 | composite key `(customer_id, month)`; monthly aggregates |

The tables join on `customer_id` only — never on dates, which are different
temporal concepts in each table (a join event, a disbursement point, a monthly
bucket).

## The 13-step cleaning framework

Each table is cleaned through the same ordered framework, applied in a
data-driven order:

1. Load & structural fixes
2. Value normalization (strip, sentinels → NaN)
3. Type conversion
4. Completeness measurement
5. Value standardization (+ type freeze)
6. Membership verification (allowed sets)
7. Ranges & cross-field validation (illegal values)
8. Outlier detection (legal-but-extreme)
9. Uniqueness / de-duplication
10. Missing-data handling
11. Derived columns
12. Uniformity & final formatting
13. Validation gate & save

Not every step does work on every table — a step that finds nothing is recorded
as a clean pass, not skipped silently.

## Pipeline architecture

Each table has a productionized module under `src/cleaning/` built from the same
parts:

- small **single-purpose functions** (one transformation each),
- a **`clean_<table>` orchestrator** that composes them,
- a **`validate_<table>` gate** (Step 13) that asserts every invariant of the
  clean data and raises on the first violation,
- a guarded **`main()`** that runs `load → clean → validate → save`.

All fixed rules (allowed category sets, legal ranges, the dataset as-of date,
file paths) live in `src/config.py`, so the schema is visible in one place and a
threshold change is a one-line edit.

The clean/validate core is decoupled from I/O: `clean_<table>` and
`validate_<table>` are pure DataFrame logic, while `main()` is the only piece
tied to disk — which keeps the pipeline reusable behind, e.g., an upload-driven
app later.

## Repository structure

```
.
├── src/
│   ├── config.py                 # paths + all cleaning schema constants
│   └── cleaning/
│       ├── customers.py
│       ├── loans.py
│       └── transactions.py
├── data/
│   ├── raw/                      # *_raw.csv  (inputs)
│   └── clean/                    # *_clean.parquet  (validated outputs)
├── docs/
│   ├── customers_cleaning_case_study.md
│   ├── loans_cleaning_case_study.md
│   └── transactions_cleaning_case_study.md
└── notebooks/                    # exploration behind each pipeline
```

## Running it

Each pipeline runs standalone and writes its clean parquet:

```bash
python -m src.cleaning.customers
python -m src.cleaning.loans
python -m src.cleaning.transactions
```

Each prints its output shape and destination, and fails loudly if its validation
gate does not pass.

## Case studies

Full step-by-step write-ups — every decision and its reasoning — live in `docs/`:

- **[Customers](docs/customers_cleaning_case_study.md)**
- **[Loans](docs/loans_cleaning_case_study.md)**
- **[Transactions](docs/transactions_cleaning_case_study.md)**

## Reconciliation (integration stage)

Cleaning each table standalone leaves a small set of problems that can only be
solved once the tables are together. These are handled in a separate
reconciliation stage, not silently patched during cleaning:

- **Referential integrity** — every `customer_id` in loans and transactions must
  exist in the clean customers table; orphans are errors.
- **Recovering parked values** — three loan rows had corrupted (negative)
  amounts, flagged rather than guessed during cleaning; they are resolved here
  against the customer's wallet inflow.

Throughout cleaning, anything that could not be resolved single-table was carried
forward **flagged**, never silently altered — a clean table may hold
known-suspect rows, but not silently-wrong ones.
