# FinSight — Conversational Product Analytics for Digital Financial Services
### Project Brief & Data Dictionary (read this before touching the data)

---

## 1) The Problem (business context)

A digital financial services company (modeled on JazzCash) runs a **mobile wallet** —
a money account tied to a phone number. Customers receive salaries, send money,
pay bills, save, buy insurance, and take small loans through it.

The product team is flying half-blind:

- Customers **leave (churn)** and nobody knows the main causes.
- **Loans default** and the approval process can't tell risky borrowers from safe ones.
- Marketing treats all customers the same because nobody has mapped **what kinds
  of customers exist**.
- Every analytical question ("which region is riskiest?") requires begging a data
  analyst and waiting days.

## 2) What we are building (the two deliverables)

**Deliverable A — the Data Science core (you as Data Scientist):**
1. Clean three raw, messy company tables into trustworthy data.
2. Explore them: find and *statistically prove* the patterns (churn drivers,
   seasonal spikes, regional risk).
3. Build three models:
   - **Churn model** — predicts which customers will leave (classification).
   - **Credit/default model** — predicts which loans will not be repaid
     (classification; must be explainable — fintech regulation).
   - **Segmentation** — discovers natural customer groups (clustering).
4. Design and analyze one **simulated A/B test** (pricing experiment).

**Deliverable B — the AI layer (you as AI Engineer):**
A chat app where a product manager types questions in plain English
("where is churn worst?") and the system: routes the question to one of YOUR
analysis functions → runs YOUR code/models → returns the numbers → an LLM writes
the business answer with a chart. The LLM never computes; it only routes and narrates.

## 3) Glossary — every technical term used in this project

| Term | Plain meaning |
|---|---|
| **Mobile wallet** | A money account on your phone number. All money in/out passes through the company's servers. |
| **Inflow** | Money ARRIVING into a wallet (salary, payments received). Company sees it in its own logs. |
| **Churn** | A customer stopping use of the service / leaving. |
| **Default** | A borrower failing to repay a loan. |
| **Credit scoring** | Estimating how safe it is to lend to someone (a number, higher = safer). |
| **KYC** | "Know Your Customer" — the signup form where users declare identity & income. Self-reported → unreliable. |
| **Proxy** | A measurable stand-in for something you can't measure. Observed inflow is a proxy for true income. |
| **Feature** | An input column a model learns from (age, tenure, inflow…). |
| **Feature engineering** | Creating better model inputs from raw columns (e.g., inflow-to-loan ratio). |
| **Target / label** | The column a model must predict (churned_12m, defaulted). |
| **Classification** | Predicting a category (will churn: yes/no). |
| **Clustering (unsupervised)** | Finding natural groups WITHOUT labels (K-Means). |
| **Sentinel value** | A fake number used to mean "missing" (age = -999). Must be cleaned. |
| **Imputation** | Filling missing values with a reasonable estimate (e.g., median). |
| **Outlier** | A value far outside plausible range (age = 250). |
| **A/B test** | Randomly split users; give one half a change; measure if it helped, statistically. |
| **p-value / significance** | How unlikely a result is under pure chance. Small p → real effect. |
| **Confidence interval** | The range a true rate plausibly lies in ("churn 7.5% ± 0.4%"). |
| **Calibration** | Whether a model's probabilities are honest (of loans scored "20% risk", ~20% should default). |
| **Model drift** | A model getting worse over time as the world changes; must be monitored. |
| **LLM router** | One LLM call that maps a user question to {which function, which arguments}. |
| **Tool** | A plain Python function YOU wrote that performs one analysis. |

## 4) The Data — three raw files (all three are messy on purpose)

### 4.1 `customers_raw.csv` — one row per customer (15,200 rows incl. ~200 duplicates)

| Column | Meaning | Known mess |
|---|---|---|
| customer_id | Unique customer key (joins all tables) | duplicated rows exist |
| age | Customer age in years | -999 sentinels; impossible 250s |
| region / city | Where the customer lives | case variants ("punjab", "PUNJAB ") |
| segment_true | ⚠ HIDDEN ANSWER KEY for clustering — drop during modeling, use only to validate | — |
| onboarding_date | When they joined the wallet | 3 date formats |
| wallet_tenure_months | Months since joining | ~5% missing |
| declared_income_band | Income they CLAIMED at KYC signup | ~12% missing; label variants ("25-50k" vs "25k-50k") |
| avg_monthly_inflow_pkr | OBSERVED money arriving into wallet / month (PKR) — the trustworthy income proxy | — |
| dependents | Family members supported | — |
| smartphone_user | Has a smartphone | mixed encodings: Yes/N/1/0/True |
| avg_monthly_txns | Average transactions per month | — |
| complaints_12m | Complaints filed in last 12 months | — |
| failed_txns_12m | Failed transactions in last 12 months | — |
| has_savings / savings_balance_pkr | Savings product usage & balance | mixed boolean encodings |
| has_insurance | Insurance product holder | mixed boolean encodings |
| credit_score | Company's current score (300–850) | — |
| churned_12m | TARGET: left in last 12 months | encoded Y/N/blank |

### 4.2 `loans_raw.csv` — one row per loan (8,050 rows incl. ~50 duplicate loan_ids)

| Column | Meaning | Known mess |
|---|---|---|
| loan_id | Unique loan key | duplicates exist |
| customer_id | Who borrowed (join key) | — |
| disbursed_date | When money was given | 2 date formats |
| purpose | nano_loan / merchant_advance / device_finance / emergency | — |
| amount_pkr | Loan size in PKR | 3 negative values (data-entry errors) |
| term_months | Repayment period | — |
| interest_rate_pct | Interest rate | some rows are text like "24.5%" |
| inflow_to_loan_ratio | loan amount ÷ monthly inflow (pre-computed feature) | — |
| defaulted | TARGET: loan not repaid | — |

### 4.3 `transactions_raw.csv` — one row per customer per month (178,200 rows; ~1% months missing)

| Column | Meaning | Known mess |
|---|---|---|
| customer_id | Join key | — |
| month | Which month (Jul-2024 … Jun-2025) | two formats: "2025-04" and "Apr-2025" |
| txn_count | Transactions that month | a few negatives |
| txn_value_pkr | Money moved that month | — |

**Note:** there is NO pre-made KPI table this time — you build the monthly KPI
summary yourself from transactions (groupby month). That's part of the work.

## 5) Hidden ground truths (the patterns your analysis MUST rediscover)

1. Churn is driven by complaints, failed transactions, and low activity;
   merchants churn least, savers most.
2. Default risk rises with inflow_to_loan_ratio, falls with credit score;
   Balochistan carries elevated risk.
3. Transaction volume spikes in Apr-2025 and Jun-2025 (Eid seasonality).
4. Four natural customer segments exist (payroll / merchant / saver / borrower) —
   your K-Means must find them WITHOUT looking at segment_true.
5. Insurance uptake is higher among savers with more dependents.
6. Declared income bands frequently contradict observed inflows (KYC lying) —
   quantify how often.

## 6) Project phases & deliverables checklist

- [ ] **Phase 0 — Cleaning:** damage report → fixes → `*_clean.csv` + cleaning log
- [ ] **Phase 1 — EDA + statistics:** distributions, churn-driver hypothesis tests,
      Eid-spike significance, regional risk with confidence intervals,
      declared-vs-observed income audit, self-built monthly KPI table
- [ ] **Phase 2 — Models:** churn (logistic + XGBoost), default (logistic + XGBoost
      + calibration curve), K-Means segmentation validated vs segment_true
- [ ] **Phase 3 — Experiment:** simulated A/B pricing test with z-test, CI, power analysis
- [ ] **Phase 4 — AI layer:** question inventory → tool catalog (YOU design it) →
      LLM router (Pydantic) → interpreter → LangGraph → Streamlit
- [ ] **Phase 5 — README** as a business case study

## 7) Scope statement (put this in your README)

This system is built for THIS company's data schema — the way real product-analytics
systems are. It handles new rows of the same shape (including the known dirt
patterns) automatically; a new schema is a new project. Generalizing across
arbitrary datasets is a product-company roadmap, not an analytics deliverable.
