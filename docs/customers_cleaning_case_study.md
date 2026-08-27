# Data Cleaning Case Study — JazzCash Customer Dataset

**Dataset:** `customers_raw.csv` — 15,200 rows, 19 columns of synthetic Pakistani mobile-wallet customer data (KYC fields, wallet behaviour, credit signals).
**Goal:** turn a deliberately-dirty raw file into a validated, correctly-typed, model-ready dataset — with every decision documented and verified.
**Deliverable:** `data/clean/customers_clean.parquet`, produced by a fully rerunnable notebook (`notebooks/01_cleaning.ipynb`).

The guiding principle throughout: **a threshold tells you when to _look_, not when to _act_.** Every fix follows the pattern **inspect → decide (with reasoning) → apply → verify.** Raw data is never mutated; all work happens on a copy, and the pipeline reruns from raw end-to-end.

---

## 0. Loading — before cleaning could even begin

The raw file failed to load at all. Two structural problems had to be solved first:

**Parse failure from whitespace padding.** The file was column-aligned with spaces (e.g. `, "May 23, 2024" ,`). A CSV quote is only honoured when it is the *first character* of a field, but the leading space pushed the quote inward, so the comma *inside* the quoted date was read as a delimiter — producing 20 fields where 19 were expected. **Fix:** `pd.read_csv(..., skipinitialspace=True)`, which strips the space after each delimiter before parsing.

**Header whitespace.** `skipinitialspace` only cleans *leading* spaces, so column names still carried trailing padding (`'age '`, `'region     '`). This silently broke every `cust["age"]`-style lookup until fixed with `cust.columns = cust.columns.str.strip()`.

**Lesson recorded:** these are *load-correctness* fixes, not data cleaning — they unblock everything downstream, so they run first, right after load.

---

## 1. Inspection — building the damage report (no changes made)

A full inspection pass before touching any value. Tools used and what each revealed:

- **`shape` / `dtypes` / `info`** — 19 columns; 10 mis-read as `object` because of dirty values or mixed formats.
- **Unique-values sweep** (`for c in object columns: print unique`) — exposed casing chaos, sentinels, and format mixing.
- **`value_counts`** — quantified each problem (systematic vs stray).
- **Numeric census** (`describe().T`) — read min/max as impossible-value detectors.

### Findings catalogued
| Column | Problem | Evidence |
|---|---|---|
| (rows) | 200 duplicate rows | 15,200 rows vs 15,000 unique customer_ids |
| `age` | sentinel `-999` (150 rows), impossible `250` (10 rows) | mean 22.5 / std 103 — corrupted; median 33 trustworthy |
| `region` | 16 case-variants of 6 real regions | `Punjab`/`punjab`/`PUNJAB`; ~28% of Punjab rows miscased (systematic) |
| `city` | 16 values — but genuinely 16 distinct cities | no case-variants; **no fix needed** |
| `onboarding_date` | 3 date formats mixed | ISO, `DD/MM/YYYY`, `Mon DD, YYYY` |
| booleans (×3) | 6 encodings each | `Yes / True / 1 / N / False / 0` |
| `declared_income_band` | variant `25k-50k` vs `25-50k`; 1,800 missing | — |
| `churned_12m` | Y/N target, 300 missing | 7.7% churn rate |
| all strings | trailing whitespace | `'Sindh      '` |

**Robustness lesson (age):** the `-999` sentinels dragged the *mean* to 22.5 and exploded *std* to 103, while the *median* (33) and *percentiles* stayed correct. Mean uses magnitudes (an extreme value enters at full size); median uses positions (an extreme value is just one slot in the sorted line). This is *why* skewed/dirty financial data is summarised with median/IQR, not mean/std.

### Plausibility check on extreme inflows (whales)
The top-10 inflow rows (up to PKR 343,700 vs a ~31,700 median) were investigated for **internal contradiction**, not mere extremeness. Apparent red flags all dissolved on inspection:
- `smartphone_user = False` with an active wallet → valid: these are **USSD** (dial-code `*786#`) customers, not app users.
- borrowers holding savings → coherent (a borrower is loan-focused, not necessarily poor).
- declared band below measured inflow → a mild stretch (band self-reported at signup; inflow measured later), not a clash.

**Verdict:** genuine **whales** — signal, not noise. **Kept and flagged**, never dropped (dropping biases every model toward the average customer).

---

## 2. Normalize sentinels & strip formatting

- **Trailing whitespace** stripped from every `object` column via a loop (`cust[col] = cust[col].str.strip()`), verified on `region`.
- **`age == -999` → `NaN`** via `.replace(-999, np.nan)`. `-999` is a *sentinel* (a deliberate "unknown" code) so the honest translation is missing, not a real age. Verified: age NaN → 150.
- `250` was **not** touched here — it is a typo, not a sentinel, and belongs to range-validation (Step 7).

---

## 3. Type conversion

**`onboarding_date` → datetime, the three-format merge.** A single `to_datetime` format can only rescue its own format and turns the rest to `NaT`. So the column was converted **three times** (one per format, `errors="coerce"`), then stitched:

```
merged = dash.combine_first(slash).combine_first(word)
```

`combine_first` keeps a value where present and fills gaps from the next — three "transparent sheets" whose holes complete each other. **Verified 0 `NaT`** (10,815 ISO + 2,685 slash + 1,500 word = 15,000, no residue). This unlocked date arithmetic, which later powered tenure recovery.

**Date-format forensics — DD/MM vs MM/DD.** The slash format was ambiguous (`10/05/2025`). Converting as day-first vs month-first and comparing survivor counts: 1,632 dates parsed *only* as day-first (first number > 12, impossible as a month), zero parsed only as month-first. **Verdict: `DD/MM/YYYY`** — matching Pakistani convention.

---

## 5. Value standardization

- **`region` 16 → 6** via an explicit `.replace({...})` mapping (not a blanket `.str.title()`, which would mangle acronyms like `KP`, `AJK-GB`). `.replace` chosen over `.map` because it leaves unlisted values *visible* rather than turning them to NaN. Verified `nunique() == 6`.
- **`city`** — confirmed 16 genuinely distinct cities, **skipped** (no fix invented for a clean column).
- **Booleans → 1/0.** `smartphone_user`, `has_savings`, `has_insurance` standardized (`Yes/True/1 → 1`, `N/False/0 → 0`) via one shared dict looped over the three columns, then `.astype(int)`. Chosen over string labels because models need numeric input. Verified values `{0,1}`, dtype `int64`.
- **`declared_income_band`** variant `25k-50k → 25-50k` folded.

### Type freeze
`region`, `city`, `segment_true` → plain `category`. `declared_income_band` → **ordered** `pd.Categorical` (bands listed low-to-high, `ordered=True`) so the natural rank (`<25k` < ... < `250k+`) is preserved for modeling and comparisons. Deferred until *after* value-cleaning so the category definitions freeze only clean labels.

**Memory: 6.3 MB → 3.2 MB (~49% reduction)** — measured with `info(memory_usage="deep")` before/after. Category dtype stores each label once plus small integer codes instead of repeating strings 15,000 times.

---

## 6. Membership constraints

Every categorical validated against its allowed set via inverted `.isin()`:
- `region` against the 6 real regions → **0 invalid**
- `segment_true` against the 4 segments → **0 invalid**

A confirmation gate proving Step 5 was complete — no unrecognized value survived.

---

## 7. Range & cross-field validation — the deepest step

### Impossible ages
`age == 250` (10 rows). Unlike `-999`, this is a *corrupted intended value*, not a code for "missing." No column in the row determines the true age (a 39-month-tenure borrower earning 44k could be any age), so no recovery is **defensible**. Guessing "probably 25" invents plausible-but-fake data — unacceptable for a credit model facing regulatory scrutiny. **Decision: nullify to NaN** — honest "unknown" beats fabrication. Age NaN: 150 → **160**.

### Tenure recovery — deterministic imputation (the standout)
`wallet_tenure_months` had 750 missing values, but tenure is **recoverable**, not guessable: tenure = time from `onboarding_date` to the dataset's reference date.

**Finding the reference ("as-of") date.** The stored tenure was measured up to whenever the dataset was frozen — *not* today's real calendar date (using `today()` would inject ~12 months of drift). The as-of date was **reverse-engineered**: for rows with both onboarding and tenure, `onboarding + tenure` points at the as-of date. Bucketing those implied dates by month:
- **2025-06: 10,354 rows**, **2025-07: 3,531 rows** → 97% of complete rows cluster in a two-month window.

**Anchor confirmed: 1 July 2025.** Validated across the full column, not assumed from one row.

**Filling the 750 blanks:** `(anchor − onboarding).dt.days / 30.44`, rounded to whole months, `fillna` into the gaps only. (`30.44` = average days per month, 365.25 ÷ 12 — the bridge between month-counts and the day-based arithmetic dates use.) **Verified 0 NaN, 0 negatives.**

**Correcting 358 _contradictory_ stored tenures.** A subtlety caught on review: `fillna` only touches NaN, so ~358 rows whose *stored* tenure was wrong (onboarding + stored landed in 2029–2031) still carried bad values. These were **detected** (implied end-date beyond a 22-July-2025 grace ceiling — the grace absorbing the 30.44 approximation wobble) and **corrected** via `.loc`, overwritten with the date-computed tenure. Re-verified: 0 overshoots.

This is the strongest fix type in the pipeline — provably wrong values replaced by provably correct ones, no guessing.

### Cross-field consistency
- **`savings_balance` vs `has_savings`** — checked for the contradiction "flag says no savings but balance > 0" → **0 rows**. Consistent.
- **`city` vs `region`** — a 16-city → region lookup mapped onto the city column, compared against the recorded region → **0 mismatches**. Every city sits in its correct region.

---

## 4 & 10. Missing-data handling — diagnose the mechanism first

Completeness re-measured on the *current* state (counts shift after dedup and sentinel conversion — `isnull()` is only honest once sentinels are real NaN).

**Mechanism diagnosis drives the treatment:**

- **`declared_income_band` (1,800 missing) → MCAR.** Verified the missing group looks identical to the present group across three dimensions: median inflow (31,700 = 31,700), regional mix (matching proportions), churn rate (7.8% vs 7.0%). Because nothing predicts the missingness, ML/model-based imputation would only fit noise — **simple imputation is correct.** Filled with the **mode** (`mode()[0]` — mode returns a Series; `[0]` takes the top value; mode is the natural fill for a *category*). A `income_band_missing` flag was added for transparency (flag created *before* filling, so it snapshots the original pattern).

- **`age` (160 missing) → median (33) + `age_missing` flag.** Median, not mean — the robust centre, resistant to the very outliers that motivated nullifying them. Flag-first, then fill.

- **`churned_12m` (300 missing) → left untouched.** It is the **target** variable; imputing a target fabricates answers. Its missing-handling is a modeling-stage decision (exclude those rows from training).

**When ML imputation _would_ apply:** only under **MAR** — when the missing column is genuinely predictable from observed columns (e.g. income missing *mostly for low-inflow customers*). Diagnosed here as MCAR, so it does not apply. And any learned imputer must be fit on the *training split only* to avoid leakage.

---

## 11. Derived columns

- **`tenure_years`** = `wallet_tenure_months / 12`, rounded. A reporting convenience (same information as months, rescaled for human readability) — adds no new predictive signal; usefulness to be confirmed at modeling.
- **`is_whale`** = `(avg_monthly_inflow_pkr > 200,000).astype(int)` → **28 whales** flagged. Gives models an explicit handle on the extreme tail rather than letting 28 outliers distort the fit for the other ~15,000.

Feature-engineering principle: derived columns inject knowledge the raw data does not state outright — but *which* actually help is validated later via feature importance, not assumed.

---

## 13. Validation gate & save

A battery of assertions encodes every belief the cleaning established — if any fails, the pipeline is wrong:

```
assert cust["customer_id"].is_unique
assert cust["age"].notna().all()
assert cust["wallet_tenure_months"].notna().all()
assert cust["declared_income_band"].notna().all()
assert cust["age"].between(18, 100).all()
assert cust["wallet_tenure_months"].between(0, 120).all()
assert cust["onboarding_date"].max() <= pd.Timestamp("2025-07-22")
assert set(cust["region"].unique()) <= {6 allowed regions}
assert set(cust["segment_true"].unique()) <= {4 allowed segments}
# booleans all in {0, 1}
```

**All checks passed.** This mirrors a schema-contract pattern (Pandera / Great Expectations turn these into declarative, reusable schemas — the production form).

**Saved to `data/clean/customers_clean.parquet`.** Parquet over CSV deliberately: CSV is plain text and would flatten the `category` and `datetime64` dtypes back to strings, discarding the type work; parquet preserves dtypes, and reloading returns the exact schema.

---

## Summary of decisions

| Issue | Decision | Why |
|---|---|---|
| 200 duplicate rows | Drop (keep first) | All exact copies (whole-row dup count == key dup count); a dupe split across train/test would leak |
| `age = -999` (150) | → NaN | Sentinel = honest "unknown" |
| `age = 250` (10) | → NaN | Typo, no defensible recovery; NaN over fabrication |
| 750 missing tenure | Compute from dates | Deterministically recoverable, not guessable |
| 358 contradictory tenures | Overwrite from dates | Provably wrong → provably correct |
| region 16→6 | `.replace` mapping | Preserves acronyms; unlisted values stay visible |
| booleans | → 1/0 int | Model-ready numeric |
| income-band missing | Mode + flag | Diagnosed MCAR → simple fill correct |
| age missing | Median + flag | Robust centre |
| churned missing | Untouched | Target — never impute |
| whales | Keep + flag | Extreme but real; signal not noise |

**Result:** 15,200 messy rows → 15,000 validated, correctly-typed, documented customers. Every fix inspected, decided with reasoning, applied, and verified. The notebook reruns from raw to clean end-to-end.
