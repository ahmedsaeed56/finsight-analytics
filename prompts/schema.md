# Data Schema

Reference for generating pandas against this project's tables, and for
narrating results. Every column name, type, and allowed value is listed. Do not
invent columns — if a question needs a column that is not here, say so.

---

## Reference figures vs the loaded file

**Every count and rate in this document describes the REFERENCE EXTRACT** —
the dataset the models were fitted on. It is not the data loaded right now.

The tools read whatever the user uploaded. A company whose real default rate is
19% must be told 19%, never 14.1% because that is what this file says.

- **Never quote a number from this document as the user's figure.** Call the
  tool and use what comes back.
- Use these figures only to explain what the *model* learned from, or to say
  how the user's data compares to it.
- Structure — column names, types, allowed values, grain rules — is stable and
  can be relied on. Only the counts and rates vary.

**Two counts in every tool return.** `n` is rows selected. `n_measured`, when
present, is rows that actually carried a value; the rest were excluded from the
figure. **Quote `n_measured`.** A scoring upload has no outcome labels, so
those customers are kept but cannot contribute to a rate — saying "7.7% of
15,000 customers churned" when the rate came from 14,700 is wrong.
`measurement_note` spells out the gap when there is one.

---

## Tables

Three feature tables, built from the current upload and held read-only.

| Variable | Grain | Use for |
|---|---|---|
| `default_df` | one row per loan | loan and default questions |
| `churn_df` | one row per customer | churn and retention questions |
| `segment_df` | one row per customer | population and behaviour questions |

**No train/test split.** The split exists only in the training path. An upload
is scored whole — there is one version of each table.

**Choosing between them.** `default_df` covers only customers who borrowed, so
it cannot describe the whole customer base. `churn_df` and `segment_df` both
cover every customer; `churn_df` adds the six-month churn window and the label,
`segment_df` carries full-panel behaviour and no outcome.

They join on `customer_id`. `default_df` also has `loan_id`, unique per row.

**Grain mismatches cannot be joined away.** A loan-level column and a
customer-level outcome describe different populations: many customers never
borrowed, so joining `default_df` onto `churn_df` silently drops them and
answers a narrower question than the one asked. "Churn rate by loan size band"
has no valid answer at the customer level. Say so rather than joining.

---

## `default_df` — one row per loan

### Loan attributes

| Column | Type | Notes |
|---|---|---|
| `loan_id` | str | unique, `L500000` format |
| `customer_id` | str | `C100000` format |
| `disbursed_date` | datetime | spans the upload's own window |
| `purpose` | category | see allowed values below |
| `amount_pkr` | float | reference range 4,000 – 400,000 |
| `term_months` | int | 1, 3, 6, or 12 |
| `inflow_to_loan_ratio` | float | loan amount ÷ monthly inflow. **The strongest default driver.** Flat below ~1.2, then climbs sharply — see `ratio_band` |
| `defaulted` | int8 (0/1) | **the target.** May be ABSENT: a loan disbursed recently has no outcome yet, and a scoring upload legitimately arrives without this column. Reference rate 14.1% |

### Pre-loan wallet behaviour

Computed from transactions strictly **before** each loan's `disbursed_date` —
what a lender would have known at the moment of the decision.

| Column | Type | Notes |
|---|---|---|
| `months_available` | int | months of history behind this loan. Not every loan has the same window |
| `total_txns` | float | transaction count in that window |
| `total_value` | float | transaction value in that window |
| `active_months` | float | months with at least one transaction |
| `average_txns_per_mon` | float | `total_txns / months_available` |
| `average_value_per_mon` | float | `total_value / months_available` |
| `active_ratio` | float | `active_months / months_available`, 0–1 |

**Use the averages, not the totals, when comparing loans.** Totals grow with
window length, so a loan with 11 months of history shows more transactions than
one with 2 for reasons that have nothing to do with the customer.

**`total_txns`, `total_value`, and `active_months` exist in more than one
table and mean different things.** Here they cover only the pre-loan window;
in `segment_df` they cover the full panel. The analytics tool resolves these
three names to `segment_df`. For the pre-loan versions, use the `average_*`
and `active_ratio` columns, which are unambiguous.

### Customer profile

`age`, `region`, `wallet_tenure_months`, `declared_income_band`,
`avg_monthly_inflow_pkr`, `dependents`, `smartphone_user`, `complaints_12m`,
`failed_txns_12m`, `has_savings`, `savings_balance_pkr`, `has_insurance`,
`credit_score`, `age_missing`, `income_band_missing`.

---

## `churn_df` — one row per customer

| Column | Type | Notes |
|---|---|---|
| `customer_id` | str | unique |
| `churned_12m` | string | **the target.** `"Y"` / `"N"`. May be ABSENT on a scoring upload, and where present may be null for some customers — those rows are kept but excluded from any rate, which is what `n_measured` reports. Reference rate 7.7% Y. Stored as pandas string dtype, not object — mapped to 1/0 before any rate is computed |

### Wallet behaviour, first half of the panel only

Features come from the first six months of the panel. The second half is the
outcome period and is deliberately excluded — using it would mean predicting a
departure from the departure itself.

| Column | Type | Notes |
|---|---|---|
| `total_counts` | float | transactions in the six-month window |
| `total_amount` | float | value in the window |
| `active_months` | float | 0–6 |
| `first` | float | transactions in months 1–3 |
| `last` | float | transactions in months 4–6 |
| `difference` | float | `last - first`. **Negative means declining activity** — the clearest early warning of churn |

Plus the same customer profile columns as `default_df`.

**No loan columns.** `purpose`, `term_months`, `amount_pkr`,
`inflow_to_loan_ratio` and `ratio_band` are loan properties and do not exist
here.

---

## `segment_df` — one row per customer

Every customer. Full panel of transaction history, no window rule — nothing is
being predicted, so there is no future to protect.

`total_txns`, `total_value`, `active_months`, plus the customer profile.

Has no `churned_12m` and no `defaulted` — for outcome questions, use the other
tables. Also has no loan columns.

---

## Band columns

Precomputed groupings of continuous columns. These are the only valid
`group_by` values for continuous quantities — grouping by `age` or
`credit_score` directly would produce hundreds of one-row groups.

Bands are **analytics-only** and are dropped before modelling, so a band never
affects a prediction.

| Band column | Source column | Labels |
|---|---|---|
| `age_band` | `age` | `18-29` · `30-40` · `41-66` |
| `complaints_band` | `complaints_12m` | `0` · `1` · `2+` |
| `failed_txns_band` | `failed_txns_12m` | `0` · `1` · `2` · `3+` |
| `dependents_band` | `dependents` | `0` · `1-2` · `3+` |
| `credit_score_band` | `credit_score` | `Q1` · `Q2` · `Q3` · `Q4` |
| `tenure_band` | `wallet_tenure_months` | `Q1` · `Q2` · `Q3` · `Q4` |
| `inflow_band` | `avg_monthly_inflow_pkr` | `Q1` · `Q2` · `Q3` · `Q4` |
| `ratio_band` | `inflow_to_loan_ratio` | `under 1.2x` · `1.2-3.5x` · `over 3.5x` — **`default_df` only** |

`Q1` is always the **lowest** quartile of the source column. For
`credit_score_band` and `tenure_band` that means `Q1` is the highest-risk
group, since default falls as both rise.

**The quartile bands are cut on the loaded file, so they are not comparable
across uploads.** `Q1` covers whatever the bottom quarter of *this* file's
credit scores happens to be. Every figure grouped by a quartile band is correct
about the loaded data, but "is Q1 getting worse month over month" cannot be
answered — the label would mean two different score ranges. Say so if asked.

**`ratio_band` is the exception, and it is not a quartile.** Its boundaries are
fixed at 1.24 and 3.5, so they mean the same thing in every file and month-over-
month comparison IS valid. They are fixed because the finding is a threshold,
not a gradient: default sits flat below roughly 1.2x monthly inflow and then
climbs sharply. Equal-sized bands blur that shape into a smooth rise, which
would support the wrong policy. The bands are therefore unequal in size by
design — in the reference extract, 3,841 / 1,277 / 1,276 loans with default
rates of 7.6% / 13.3% / 34.5%.

---

## Allowed values

Filters must use these exact strings. They are case-sensitive.

```
region Punjab · Sindh · KP · Islamabad · AJK-GB · Balochistan
purpose nano_loan · merchant_advance · device_finance · emergency
declared_income_band <25k · 25-50k · 50-100k · 100-250k · 250k+ (ordered)
churned_12m Y · N
term_months 1 · 3 · 6 · 12
```

`declared_income_band` is an **ordered** category, so comparisons work:
`df[df["declared_income_band"] > "50-100k"]` is valid.

**Region counts are uneven, and which regions are thin depends on the file.**
Any group below 400 rows is flagged automatically, because that is roughly the
sample the A/B power analysis showed is needed to detect a moderate effect.
Report a flagged figure with its count beside it. In the reference extract this
fires on Balochistan and AJK-GB.

A value outside these lists can still be counted and grouped — it is real data
— but it cannot be scored, because the model never saw it. Report it as
present and unscorable rather than silently ignoring it.

---

## Risk bands

Model probabilities are cut into three bands. The cut points are a **policy
choice** set in `config.py`, not a model output — they can be changed without
retraining, and they are the same for every file.

These are unrelated to the band columns above: those group a *measured*
column, these group a *predicted* probability.

The rates below are what the reference extract produced at these cuts. **They
describe the model's training population, not the loaded file.** For the
loaded file's band composition, call the tool.

### Default risk

| Band | Probability | Share of loans | Observed default rate | Share of volume |
|---|---|---|---|---|
| low | < 0.30 | 32.5% | 3.3% | 15% |
| medium | 0.30 – 0.60 | 49.9% | 13.8% | 36% |
| high | > 0.60 | 17.6% | 32.3% | 49% |

Current policy: **low** → standard approval · **medium** → standard terms ·
**high** → manual review, reduced limit, or risk-based pricing.

**The high band holds half of all disbursed volume.** This is why the model is
not used as an accept/decline gate: no threshold that catches a useful share of
defaults leaves enough volume to satisfy the 15% guardrail. Bands work because
they vary *treatment*, not access — nobody is declined.

### Churn risk

| Band | Probability | Share of customers | Observed churn rate | Share of inflow |
|---|---|---|---|---|
| low | < 0.40 | 42.5% | 2.2% | 47% |
| medium | 0.40 – 0.70 | 48.5% | 10.0% | 45% |
| high | > 0.70 | 9.0% | 21.1% | 8% |

Current policy: **high** → retention outreach · **medium** → monitor ·
**low** → no action.

**Churn risk sits in the least valuable customers.** The high band holds 24.7%
of all churners but under 8% of monthly inflow. A campaign sized on customer
count would spend most of its budget defending revenue that was never at risk.

---

## What this data cannot answer

State these plainly rather than working around them.

**No calendar dates on churn.** `churned_12m` is a flag over the whole panel
with no date attached. "How many churned last month" is not answerable. "What
share of customers churned during the observation period" is.

**No repayment timeline.** `defaulted` is a final outcome. There is no data on
when a borrower stopped paying or how many instalments were made.

**Transactions are monthly, not daily.** A month appears as `2024-07-01`, which
is a label for the whole of July, not a transaction on the 1st. Within-month
timing questions cannot be answered.

**Labels may be absent entirely.** A scoring upload has no `defaulted` and no
`churned_12m`, because those outcomes have not happened yet. Every question
about an observed rate is unanswerable on such a file — predictions are still
available. Say which one the user is asking for.

**Quartile bands are not comparable across uploads.** See the band section.

**One loan per customer** in the reference extract, so repeat-borrower
behaviour was never modelled.

**Loan attributes cannot describe the customer base.** Any question combining
a loan property with a customer outcome covers borrowers only. There is no way
to widen it.

**Everything is association, not causation.** The models find correlations. Say
"associated with", never "causes" or "leads to". The only causal claim
available is from the A/B test, where assignment was randomised.

**The reference extract is synthetic.** Relationships were designed by a
generator. Findings from it demonstrate method, not real market behaviour. A
user's uploaded data is real; the model fitted to it is not.

---

## Known quirks

**Regional coefficients are measured against AJK-GB.** One-hot encoding drops
the first category alphabetically, so every region coefficient reads as
"relative to AJK-GB" — the smallest region and the lowest-default group in the
reference extract. That inflates all of them. Compare regions to each other,
never read the raw value as absolute risk.

**`avg_monthly_txns` is not in these tables.** The stored profile field
disagreed with the transaction panel by roughly 6x and could not be reconciled
to any period. It was dropped. Computed transaction columns are the only
activity figures available.

**`interest_rate_pct` is not in `default_df`.** It was constructed as a
function of `credit_score` (correlation −0.76) and carries no independent
information.

**Some churn feature windows are contaminated.** Customers who left during the
first half of the panel have dead months inside their feature window. A
heuristic to detect and remove them was tested and rejected — at its best
threshold it was correct for only 64% of flagged customers.

**The model never saw a very new customer.** Reference tenure runs from about
9 months upward, so a prediction for someone newer is an extrapolation: a
number comes back, its direction is probably right, its magnitude is unverified.

**The model only ever saw loans that were disbursed.** It never saw an
application that was declined, so it ranks the kind of applicant the company
already lends to and says nothing about those the current process turns away. 