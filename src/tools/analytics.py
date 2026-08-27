import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from src.tools.errors import ToolError
from src.tools.dataset import _frame 


METRICS = {
    # --- outcomes ---
    "defaulted":              "default",
    "churned_12m":            "churn",

    # --- loan attributes (default table only) ---
    "amount_pkr":             "default",
    "term_months":            "default",
    "inflow_to_loan_ratio":   "default",

    # --- pre-loan wallet behaviour (bounded by disbursed_date) ---
    "months_available":       "default",
    "average_txns_per_mon":   "default",
    "average_value_per_mon":  "default",
    "active_ratio":           "default",

    # --- churn-window behaviour (months 1-6 only) ---
    "total_counts":           "churn",
    "total_amount":           "churn",
    "first":                  "churn",
    "last":                   "churn",
    "difference":             "churn",

    # --- full 12-month wallet behaviour ---
    "total_txns":             "segment",
    "total_value":            "segment",
    "active_months":          "segment",

    # --- customer profile (identical across tables; read from segment) ---
    "age":                    "segment",
    "credit_score":           "segment",
    "wallet_tenure_months":   "segment",
    "avg_monthly_inflow_pkr": "segment",
    "savings_balance_pkr":    "segment",
    "dependents":             "segment",
    "complaints_12m":         "segment",
    "failed_txns_12m":        "segment",
    "has_savings":            "segment",
    "has_insurance":          "segment",
    "smartphone_user":        "segment",
}

GROUP_BYS = {
    # --- categorical ---
    "region",                 # 6 levels
    "purpose",                # 4 levels — DEFAULT TABLE ONLY
    "declared_income_band",   # 5 levels, ordered
    "term_months",            # 4 levels — DEFAULT TABLE ONLY

    # --- low-cardinality numeric ---
    "has_savings",            # 2
    "has_insurance",          # 2
    "smartphone_user",        # 2
    "dependents",             # ~9 levels
    "complaints_12m",         # ~7 levels

    # --- precomputed bands (analytics only) ---
    "age_band",               # 3 levels
    "complaints_band",        # 3 levels
    "failed_txns_band",       # 4 levels
    "dependents_band",        # 3 levels
    "credit_score_band",      # 4 quartiles
    "tenure_band",            # 4 quartiles
    "inflow_band",            # 4 quartiles
    "ratio_band",             # 3 bands — DEFAULT TABLE ONLY

    # --- WHEN THE LOAN WAS GIVEN — DEFAULT TABLE ONLY -------------------
    # Derived at query time from disbursed_date. See _DERIVED below for why
    # this exists and, more importantly, what it does NOT enable.
    "disbursed_month",
}

AGGFUNCS = {"mean", "sum", "count", "median", "min", "max", "std", "nunique"}

# Below this a rate moves a lot on a handful of rows. 400 comes from the A/B
# power analysis: detecting an 8-point difference at 80% power needed 389 rows
# per arm, so a group smaller than that cannot reliably show an effect of the
# size this project cares about. Flagged in the return, never blocked — the
# number is real, it just needs saying out loud.
#
# An absolute row count, so it stays meaningful on any file size. On the
# reference extract it fires on Balochistan (304) and AJK-GB (337); which
# groups it catches in an upload depends on that upload.
SMALL_GROUP = 400

# Grid cells are much smaller than whole groups, so SMALL_GROUP's 400 would
# flag nearly every cell and stop meaning anything. 50 is where a cell stops
# being readable: at the reference extract's ~14% default rate it expects 7
# events, just above the minimum of 5 that chi-square's approximation needs.
# Below it a single loan moves the rate by two points or more — the H2xH3
# grid's 35-loan cell is the reference case.
SMALL_CELL = 50

# Properties of a LOAN, not of a person: a customer has no purpose or term,
# and no loan-to-inflow ratio until they borrow. Counting these means the
# loans table. Everything else in GROUP_BYS is customer profile, which lives
# identically in all three tables — read from segment, since it holds every
# customer rather than only the labelled ones the churn table carries.
#
# disbursed_month joins them: a customer has no disbursement date either.
_LOAN_ONLY = {"purpose", "term_months", "ratio_band", "disbursed_month"}

# ==========================================================================
#  DERIVED GROUPING COLUMNS
# ==========================================================================
# Columns that do not exist in any stored frame and are computed when a
# question asks for them. One entry so far.
#
# WHY disbursed_month EXISTS, AND WHAT IT IS NOT
# ---------------------------------------------
# The loans table carries disbursed_date — when the money went out. It does
# NOT carry an outcome date: `defaulted` is a terminal flag with no timestamp,
# so nothing records WHEN a loan went bad.
#
# That distinction is the whole of it, and collapsing the two is how this
# system spent months refusing every question containing the word "time":
#
#   "how has the default rate changed over the last six months?"
#       -> UNANSWERABLE. Needs outcome dates. There are none.
#
#   "how many loans were disbursed each month?"
#       -> ANSWERABLE, and always was. disbursed_date has been sitting in the
#          frame the entire time.
#
#   "do loans disbursed in March default more than loans disbursed in June?"
#       -> ANSWERABLE, and it is VINTAGE ANALYSIS — a real technique, not a
#          workaround. Each month is a cohort of loans, and the flag says how
#          that cohort ended up. It is not a trend line of the default rate;
#          it is a comparison between cohorts.
#
# ONE TRAP WORTH NAMING. Later vintages have had less time to go bad. A loan
# disbursed the month before the file was cut may simply not have matured, so
# a falling rate at the right-hand end is maturity, not improvement. That is
# the same panel-attrition artefact H9 hit, and the narrator is told about it.
#
# DERIVED AT QUERY TIME, NOT AT FEATURE-BUILD TIME. One derivation, one place,
# no migration, and no stored column that a re-upload could leave stale. The
# cost is a to_period call per query on a column that is already datetime —
# unmeasurable against the model call that follows.
_DERIVED = {
    "disbursed_month": "disbursed_date",
}

# Grouping columns whose natural order is chronological, not by value. A
# month list sorted by default rate is unreadable — 2025-03 above 2024-11
# above 2025-01 tells you nothing about the shape of anything. So these keep
# their own order and IGNORE the `sort` flag.
_CHRONOLOGICAL = {"disbursed_month"}

# The only metrics compare_groups can test. It counts EVENTS and NON-EVENTS
# for a chi-square table, which is only meaningful on a yes/no outcome. Named
# here so the error message can point at them.
_OUTCOME_METRICS = ("defaulted", "churned_12m")

# CUSTOMER attributes that may be merged onto the LOANS frame when a question
# needs both — "default rate by savings status" reads a loan outcome and a
# customer trait, and neither table holds both.
#
# WHY AN ALLOWLIST RATHER THAN "ANY MISSING COLUMN"
# -------------------------------------------------
# Three kinds of column must never be merged this way:
#
#   OUTCOMES. churned_12m describes a customer, defaulted describes a loan.
#   Merging churn onto loans would give a customer with three loans three
#   churn rows, and the rate would count them three times. Merging default
#   UP to customers is worse: a customer with one bad loan out of three is
#   neither clearly a defaulter nor clearly not, and choosing for them is a
#   judgement the tool has no business making.
#
#   WINDOWED BEHAVIOUR. total_txns, total_value and active_months exist in
#   BOTH the segment and churn frames with DIFFERENT windows behind them —
#   full panel in one, first six months in the other. A merge would silently
#   pick one and the answer would look identical either way.
#
#   ANYTHING PER-LOAN. Nothing on the loans side needs merging; it is already
#   there.
#
# What is left is stable customer TRAITS. One customer has one value, it does
# not change between their loans, and copying it onto each of their loans
# states nothing that was not already true.
_JOINABLE = {
    "has_savings",
    "has_insurance",
    "smartphone_user",
    "declared_income_band",
    "dependents",
    "complaints_12m",
    "age_band",
    "credit_score_band",
    "tenure_band",
    "inflow_band",
    "complaints_band",
    "failed_txns_band",
    "dependents_band",
}

# The column both frames are keyed on. Named once so the merge and its guard
# cannot disagree about it.
_JOIN_KEY = "customer_id"


def validate(metric=None, group_by=None, aggfunc="mean", filters=None):
    """Check the arguments against the whitelists.

    Runs before any pandas. Raises ToolError on the first bad argument;
    returns None if everything passes.

    `metric` is optional because band_distribution counts rows rather than
    measuring anything, so it has no metric to check. Every other tool
    declares metric as a required positional argument, so a missing one is
    caught by Python before this function is reached.

    Only NAMES are checked here. Three things need the dataframe and are
    therefore checked later: whether a column exists in the table the metric
    selected (ratio_band passes this whitelist but is not in the churn
    table), whether a filter's VALUE is real ("Karachi" passes as a region
    key and is not a region), and whether a metric is the right SHAPE for the
    tool asking (age is a real metric and compare_groups cannot use it).
    """
    if metric is not None and metric not in METRICS:
        raise ToolError(
            f"'{metric}' is not an available metric. "
            f"Available: {sorted(METRICS)}"
        )

    if group_by is not None and group_by not in GROUP_BYS:
        raise ToolError(
            f"cannot group by '{group_by}'. "
            f"Available: {sorted(GROUP_BYS)}"
        )

    if aggfunc not in AGGFUNCS:
        raise ToolError(
            f"'{aggfunc}' is not a supported aggregation. "
            f"Available: {sorted(AGGFUNCS)}"
        )

    if filters is not None:
        for key in filters:
            if key not in GROUP_BYS:
                raise ToolError(
                    f"cannot filter on '{key}'. "
                    f"Available: {sorted(GROUP_BYS)}"
                )


def _add_derived(dataframe, column):
    """Compute a derived grouping column onto the frame.

    Currently only disbursed_month, from disbursed_date. Written as a lookup
    rather than an if-branch so a second derived column is one dict entry
    rather than another special case threaded through _prepare.

    Raises rather than returning silently when the source column is absent.
    An upload whose loans file has no disbursed_date genuinely cannot answer
    a by-month question, and saying so names the missing column — which is
    something the router can act on, where an empty result is not.
    """
    source = _DERIVED[column]

    if source not in dataframe.columns:
        raise ToolError(
            f"'{column}' is worked out from '{source}', and this file has no "
            f"'{source}' column — so there is no way to tell when each loan "
            f"was given out. Comparisons across regions, products or customer "
            f"types do not need it."
        )

    dates = pd.to_datetime(dataframe[source], errors="coerce")

    if dates.isna().all():
        raise ToolError(
            f"'{source}' holds no readable dates in this file, so '{column}' "
            f"cannot be worked out from it."
        )

    # to_period("M") then str: "2025-03". Sorts correctly as text because the
    # year leads and the month is zero-padded, which is why this format is
    # used rather than "Mar 2025".
    dataframe = dataframe.copy()
    dataframe[column] = dates.dt.to_period("M").astype(str)

    # A row with an unreadable date becomes the string "NaT", which would
    # appear as a group named NaT sitting between real months. Dropped, and
    # not silently — the count goes back to the caller.
    bad = int((dataframe[column] == "NaT").sum())
    if bad:
        dataframe = dataframe[dataframe[column] != "NaT"]

    note = None
    if bad:
        note = (
            f"{bad:,} rows excluded — no readable '{source}', so the month "
            f"they were given out is unknown"
        )

    return dataframe, note


def _as_numeric(series):
    """Return a Series safe to aggregate as a rate.

    ``churned_12m`` is stored as "Y"/"N" strings, and mean() on strings
    errors outright. Every other metric is already numeric and passes
    through untouched — including ``defaulted``, which the feature build
    saved as int8 rather than bool.

    The Y/N test looks at the VALUES, not the dtype: pandas may store that
    column as object, string, or category depending on how the parquet was
    written, and an earlier dtype check silently failed for exactly that
    reason. The bool branch is defensive — nothing currently hits it.
    """
    if set(series.dropna().unique()) <= {"Y", "N"}:
        return series.map({"Y": 1, "N": 0})
    if series.dtype == bool:
        return series.astype(int)
    return series


def _is_binary(series):
    """True when every non-null value is 0 or 1.

    Checked on VALUES rather than on a list of metric names, so a 0/1 column
    in someone else's upload works without this file being edited. Run after
    _as_numeric, so Y/N has already become 1/0.
    """
    return set(series.dropna().unique()) <= {0, 1}


def _merge_customer_column(dataframe, frame, column):
    """Bring one customer TRAIT onto the loans frame.

    Answers the class of question neither table can answer alone: the outcome
    is a property of the LOAN and the split is a property of the PERSON.
    "Default rate for customers with savings" needs `defaulted` from one frame
    and `has_savings` from the other.

    A LEFT merge on customer_id. One customer has one value for these traits,
    so a borrower with three loans gets the same flag on all three — which is
    a true statement about each of those loans, not a duplication of anything.

    ORPHAN LOANS ARE DROPPED, AND SAID SO. The pipeline deliberately KEEPS
    loans whose customer is missing from the customers file — the loan is real
    and analytics answers correctly on it. But it has no customer to inherit a
    trait from, so after the merge its value is null and it cannot be grouped.
    Dropping it silently would leave n_total counting rows that n_per_group
    does not, and the narrator has no way to notice that. So the count comes
    back and lands in the filters field, where it is visible.

    Returns
    -------
    (dataframe, note) — the merged frame, and a string describing dropped
    rows, or None when nothing was dropped.
    """
    if column not in _JOINABLE:
        raise ToolError(
            f"'{column}' describes a customer and cannot be combined with "
            f"loan-level figures. Customer TRAITS that can be combined: "
            f"{sorted(_JOINABLE)}. Outcomes cannot: churn is recorded per "
            f"customer and default per loan, so one customer with several "
            f"loans has no single answer, and the tool will not choose one "
            f"for them."
        )

    if _JOIN_KEY not in dataframe.columns:
        raise ToolError(
            f"cannot combine '{column}' with this table — the {frame} table "
            f"has no {_JOIN_KEY} column to match customers on."
        )

    customers = _frame("segment")

    if column not in customers.columns:
        raise ToolError(
            f"'{column}' is not a column in the customer table either, so "
            f"there is nothing to combine. Available group-by columns: "
            f"{sorted(GROUP_BYS)}"
        )
    if _JOIN_KEY not in customers.columns:
        raise ToolError(
            f"cannot combine '{column}' — the customer table has no "
            f"{_JOIN_KEY} column to match on."
        )

    before = len(dataframe)

    # Only the key and the one column. Pulling the whole customer frame in
    # would collide with every same-named column on the loans side — the
    # window-sensitive transaction columns above all.
    dataframe = dataframe.merge(
        customers[[_JOIN_KEY, column]],
        on=_JOIN_KEY,
        how="left",
    )

    unmatched = int(dataframe[column].isna().sum())
    if unmatched:
        dataframe = dataframe[dataframe[column].notna()]

    if dataframe.empty:
        raise ToolError(
            f"no rows survived combining '{column}' from the customer table — "
            f"none of the {before:,} rows matched a customer. Check that the "
            f"two files describe the same population."
        )

    note = None
    if unmatched:
        note = (
            f"{unmatched:,} of {before:,} rows excluded — no matching "
            f"customer record to read '{column}' from"
        )

    return dataframe, note


def _prepare(metric=None, group_cols=(), aggfunc="mean", filters=None):
    """Shared setup for every Tier 1 tool.

    Validates the names, selects and copies the right table, derives any
    computed column the request needs, brings in any customer trait it needs,
    checks the filters, and applies them. Returns the ready frame, the table
    name, and the filters that were applied.

    `metric` is optional: with one, the table comes from METRICS. Without
    one — band_distribution, which counts rows rather than measuring — the
    table is chosen from the request itself, since purpose, term_months,
    ratio_band and disbursed_month are properties of a loan rather than of a
    person.

    `group_cols` is however many grouping columns the caller has: none, one,
    or two. Nones are ignored, so an optional group_by needs no special
    handling at the call site.

    ORDER OF THE THREE COLUMN STEPS, AND WHY IT IS THIS ORDER
    ---------------------------------------------------------
    DERIVE first, MERGE second, FILTER third.

    Derive before merge because a derived column is computed from a column
    already on this frame, and merging first would only make the frame bigger
    for no reason.

    Both before filtering, because a filter may name either kind:
    filters={"disbursed_month": "2025-03"} needs the column to EXIST before
    its value can be checked, and the same is true of
    filters={"has_savings": 1} on the loans frame.

    WHERE THE TWO TABLES MEET
    -------------------------
    A grouping column missing from the selected frame used to be a refusal.
    Now, if it is a stable customer TRAIT, it is merged in on customer_id —
    which is what makes "default rate by savings status" answerable at all.
    See _merge_customer_column for what may be merged and what may not.

    This is also the single place the dataset is read, which is why pointing
    the tools at an upload was one edit rather than four.
    """
    validate(metric=metric, aggfunc=aggfunc, filters=filters)

    cols = [c for c in group_cols if c is not None]
    for col in cols:
        validate(group_by=col)

    if metric is not None:
        frame = METRICS[metric]
    else:
        # The filter columns count too, not just the grouping ones:
        # filtering to nano_loan borrowers is a loans question even when the
        # split is by region.
        wanted = set(cols)
        if filters is not None:
            wanted |= set(filters)
        frame = "default" if wanted & _LOAN_ONLY else "segment"

    dataframe = _frame(frame).copy()

    applied = []

    # Grouping columns and filter columns both count: filtering to savings
    # holders needs has_savings present just as much as grouping by it does.
    needed = list(cols)
    if filters is not None:
        needed += [k for k in filters if k not in needed]

    # --- 1. derive whatever is computed rather than stored -----------------
    for col in needed:
        if col in _DERIVED and col not in dataframe.columns:
            if frame != "default":
                raise ToolError(
                    f"'{col}' says when a LOAN was given out, and this "
                    f"question is about customers. One customer can hold "
                    f"several loans given out in different months, so there "
                    f"is no single '{col}' for a person. Ask about loans "
                    f"instead."
                )
            dataframe, note = _add_derived(dataframe, col)
            if note:
                applied.append(note)

    # --- 2. bring in whatever this request needs from the other table ------
    for col in needed:
        if col in dataframe.columns:
            continue

        # A LOAN property asked of the customer table cannot be merged the
        # other way: a customer with three loans has three purposes, and
        # copying one onto them would invent an answer.
        if col in _LOAN_ONLY:
            raise ToolError(
                f"'{col}' is a property of a LOAN, and this question is about "
                f"customers. One customer can hold several loans with "
                f"different values, so there is no single '{col}' for a "
                f"person. Ask about loans instead, or group by a customer "
                f"trait."
            )

        dataframe, note = _merge_customer_column(dataframe, frame, col)
        if note:
            applied.append(note)

    # --- 3. filters --------------------------------------------------------
    if filters is not None:
        for key, value in filters.items():
            values = dataframe[key].unique()
            if value not in values:
                raise ToolError(
                    f"'{value}' is not a value of '{key}'. "
                    f"Available: {sorted(values)}"
                )
            dataframe = dataframe[dataframe[key] == value]
            applied.append(f"{key}={value}")

        if dataframe.empty:
            raise ToolError(
                f"no rows match {', '.join(applied)}. "
                f"Each filter is valid on its own, but the combination is "
                f"empty."
            )

    return dataframe, frame, applied


def vocabulary():
    """The metric names, group-by columns, and allowed values in the LOADED data.

    WHAT THE ROUTER FILLS params FROM.

    schema.md describes the reference extract — six regions, four purposes, the
    band definitions. That is stable prose and it explains the rules, but it
    cannot be current. A company uploading a file with a seventh region would
    have a router that never routes to it, because the document says there are
    six.

    So the rules come from schema.md and the VALUES come from here, read off
    the frames that are actually loaded. Same split as freeform.py's
    schema_summary: static explanation in a file, live facts generated.

    Only low-cardinality columns get their values listed. credit_score has
    hundreds and listing them would be noise; region has six and listing them
    is what stops the router filtering on "punjab" when the column holds
    "Punjab".

    DERIVED COLUMNS ARE COMPUTED HERE TOO. disbursed_month does not exist in
    any stored frame, so the plain "is it in the columns" test would skip it
    silently and the router would never learn its range. A file spanning more
    than twelve months lists its first and last instead of every value —
    enough for the router to know the span without a wall of month strings.

    Returns
    -------
    A plain string for the router prompt.
    """
    lines = [
        "METRICS (valid values for the `metric` argument):",
        "  " + ", ".join(sorted(METRICS)),
        "",
        "OUTCOME METRICS (the only ones compare_groups can test):",
        "  " + ", ".join(_OUTCOME_METRICS),
        "",
        "GROUP-BY AND FILTER COLUMNS:",
        "  " + ", ".join(sorted(GROUP_BYS)),
        "",
        "ALLOWED VALUES — filters must match these exactly, including case:",
    ]

    for column in sorted(GROUP_BYS):
        # The same table-selection rule the tools use: loan properties live
        # only in the default frame, everything else describes customers and
        # is read from segment, which holds every one of them.
        frame = "default" if column in _LOAN_ONLY else "segment"
        dataframe = _frame(frame)

        # Derived columns are absent from the stored frame by definition, so
        # they are computed here rather than skipped.
        if column in _DERIVED:
            try:
                dataframe, _ = _add_derived(dataframe, column)
            except ToolError:
                # The source column is missing from this upload, so the
                # derived one genuinely does not exist. Leaving it off the
                # values list is correct; the tools raise the same error with
                # the same explanation if anything routes to it.
                continue

        if column not in dataframe.columns:
            continue

        values = dataframe[column].dropna().unique()

        # A time column with a long span gets its ENDS rather than its
        # middle. "18 months, 2024-01 to 2025-06" tells the router what it
        # needs — that months are valid filter values and roughly which ones
        # — without eighteen lines of noise.
        if column in _CHRONOLOGICAL and len(values) > 12:
            ordered = sorted(str(v) for v in values)
            lines.append(
                f"  {column:<22} {len(ordered)} months, "
                f"{ordered[0]} to {ordered[-1]} (format YYYY-MM)"
            )
            continue

        # Above this a list is noise rather than help — the same judgement
        # MAX_LISTED_VALUES makes in the Tier 3 schema summary.
        if len(values) > 12:
            continue

        # Categories sort in their DEFINED order, which for
        # declared_income_band is the meaningful one rather than alphabetical.
        if isinstance(dataframe[column].dtype, pd.CategoricalDtype):
            listed = [str(v) for v in dataframe[column].cat.categories
                      if v in set(values)]
        else:
            listed = [str(v) for v in sorted(values)]

        lines.append(f"  {column:<22} {', '.join(listed)}")

    return "\n".join(lines)


def aggregate_metric(
    metric,
    group_by=None,
    aggfunc="mean",
    filters=None,
    sort=True,
    limit=None,
):
    """One number, or one number per group.

    Returns structured facts only — the result, the row counts behind it,
    and any caveat. Never prose: the narration layer turns flags into
    sentences.

    TWO DIFFERENT COUNTS
    --------------------
    `n` is rows selected. `n_measured` is rows that actually carried a value.
    They diverge on an upload: a scoring file has no churn label, so the
    builders keep those customers (drop_unlabeled=False) and mean() skips
    them. Reporting only `n` would have the narrator say "7.7% of 15,000
    customers churned" when the rate came from 14,700 — and the narrator has
    no way to detect that itself.

    The measured keys appear ONLY when they differ, so a fully labelled
    table returns exactly what it always did.

    SORTING BY VALUE IS WRONG FOR A TIME COLUMN
    -------------------------------------------
    `sort=True` orders groups by their figure, highest first, which is what
    "which region is worst" wants. For disbursed_month it destroys the only
    thing that makes the answer readable: a month list must run in date order,
    or nobody can see a shape in it. So a chronological column keeps its own
    order regardless of `sort`, and the return says so rather than leaving the
    narrator to wonder why its request was ignored.
    """
    dataframe, frame, applied = _prepare(metric, (group_by,), aggfunc, filters)

    checked = _as_numeric(dataframe[metric])

    if group_by is None:
        n_rows = int(len(dataframe))
        n_measured = int(checked.notna().sum())

        result = {
            "metric": metric,
            "aggfunc": aggfunc,
            "table": frame,
            "filters": applied or None,
            "result": round(float(checked.agg(aggfunc)), 4),
            "n": n_rows,
            # Powered by what backs the RATE, not by rows selected: 400 rows
            # where 300 are unlabelled is a 100-row estimate.
            "small_group_warning": n_measured < SMALL_GROUP,
        }

        if n_measured != n_rows:
            result["n_measured"] = n_measured
            result["measurement_note"] = (
                f"{n_rows - n_measured:,} of {n_rows:,} rows have no value "
                f"for '{metric}' and are excluded from the figure. The "
                f"result describes the {n_measured:,} rows that do."
            )
        return result

    # observed=True matters: several group-bys are category dtype, so without
    # it pandas emits a row for every level even when filtering left it empty.
    group_result = checked.groupby(dataframe[group_by], observed=True)
    results_total = group_result.agg(aggfunc)

    # size() counts every row in the group; count() counts only those with a
    # value. Identical on a labelled table, different on a scoring upload.
    counts = group_result.size()
    measured = group_result.count()

    chronological = group_by in _CHRONOLOGICAL

    if chronological:
        # Date order, always. The YYYY-MM format sorts correctly as text.
        results_total = results_total.sort_index()
    elif sort:
        results_total = results_total.sort_values(ascending=False)

    if limit:
        results_total = results_total.head(limit)

    # counts still holds every group in the original order, so after sorting
    # and trimming it no longer lines up with results. Reindexing forces it
    # to match, or the return dict pairs one group's rate with another
    # group's row count. measured needs the same treatment for the same
    # reason.
    counts = counts.reindex(results_total.index)
    measured = measured.reindex(results_total.index)

    thin = [str(g) for g, n in measured.items() if n < SMALL_GROUP]

    result = {
        "metric": metric,
        "aggfunc": aggfunc,
        "group_by": group_by,
        "table": frame,
        "filters": applied or None,
        "result": {
            str(k): round(float(v), 4) for k, v in results_total.items()
        },
        "n_per_group": {str(k): int(v) for k, v in counts.items()},
        "n_total": int(len(dataframe)),
        # Named, not a blanket flag — the narrator should caveat the specific
        # thin figure rather than the whole answer.
        "small_groups": thin or None,
        # The book-level figure, so a narrator comparing one group to the
        # whole doesn't have to make a second call.
        "overall": round(float(checked.agg(aggfunc)), 4),
    }

    if chronological:
        # Two flags, because they carry two different warnings.
        #
        # `ordering` explains why a `sort` request was ignored.
        #
        # `cohort_note` is the one that matters. Each month is a COHORT of
        # loans, not a point on a trend line, and the later cohorts have had
        # less time to go bad. A rate that falls at the right-hand end may be
        # immaturity rather than improvement — the same panel-attrition
        # artefact the H9 work hit. The narrator cannot see this from the
        # numbers, so it is stated here.
        result["ordering"] = "chronological"
        result["cohort_note"] = (
            f"Each group is the set of loans given out in that month, and "
            f"'{metric}' is their eventual outcome — so this compares "
            f"cohorts rather than tracking a rate through time. Loans given "
            f"out nearer the end of the file have had less time to go bad, "
            f"so a lower figure in the last month or two may be immaturity "
            f"rather than improvement."
        )

    if not measured.equals(counts):
        result["n_measured_per_group"] = {
            str(k): int(v) for k, v in measured.items()
        }
        result["n_measured_total"] = int(checked.notna().sum())
        result["measurement_note"] = (
            f"Some rows have no value for '{metric}'. n_per_group counts "
            f"rows selected; n_measured_per_group counts those behind the "
            f"figure."
        )

    return result 


def compare_groups(metric, group_by, groups=None, filters=None):
    """Compare an OUTCOME rate across groups, with a significance test.

    `groups` controls what is compared:
      None                    -> every level of the column
      ["Punjab", "Sindh"]     -> those two only, others excluded
      ["Balochistan"]         -> that one against everything else pooled

    Returns rates, counts, effect size and a chi-square p-value. The p-value
    is withheld (None) when an expected cell count falls below 5, since the
    test's approximation does not hold there — the rates stay valid.

    OUTCOMES ONLY, AND WHY
    ----------------------
    This builds a table of EVENTS and NON-EVENTS and hands it to chi-square,
    which is only meaningful when every row either did the thing or did not.
    Handed a measurement instead — age, loan amount, credit score — `events`
    becomes a SUM of that measurement, `rows - events` goes deeply negative,
    and scipy raises "all values in observed must be nonnegative" several
    frames from the actual mistake.

    So the shape is checked here, in a message the router can act on: it names
    the outcomes that work and points at aggregate_metric, which compares
    AVERAGES across groups and is what a question about age actually wants.
    """
    dataframe, frame, applied = _prepare(metric, (group_by,), filters=filters)

    if groups is not None and not groups:
        raise ToolError(
            f"'groups' was given as an empty list. Pass at least one value of "
            f"'{group_by}' to compare, or omit 'groups' entirely to compare "
            f"all levels. One value means that group against all others "
            f"pooled."
        )

    if groups is not None:
        levels = dataframe[group_by].unique()
        for val in groups:
            if val not in levels:
                raise ToolError(
                    f"'{val}' is not a value of '{group_by}'. "
                    f"Available: {sorted(levels)}"
                )

    if groups is None:
        group_col = dataframe[group_by]
    elif len(groups) >= 2:
        mask = dataframe[group_by].isin(groups)
        dataframe = dataframe[mask]
        group_col = dataframe[group_by]
    else:
        # One name means "this group vs everything else". Labels rather than
        # True/False so the narrator cannot mix up which side is which.
        group_col = pd.Series(
            np.where(dataframe[group_by] == groups[0], groups[0], "rest"),
            index=dataframe.index,
        )

    # Built after the branch, not before: the two-or-more case narrows the
    # frame, and a Series derived earlier would no longer line up with it.
    checked = _as_numeric(dataframe[metric])

    # THE SHAPE CHECK. Before any counting, because counting a measurement is
    # what produces the negative cells. See the docstring.
    if not _is_binary(checked):
        raise ToolError(
            f"compare_groups tests whether an OUTCOME happens more in one "
            f"group than another, so it needs a yes/no column — it counts how "
            f"many did the thing and how many did not. '{metric}' is a "
            f"measurement rather than an outcome, so there is nothing to "
            f"count. Outcomes available: {list(_OUTCOME_METRICS)}. "
            f"To compare the AVERAGE {metric} between groups instead, use "
            f"aggregate_metric with group_by='{group_by}'."
        )

    grouped = checked.groupby(group_col, observed=True)
    events = grouped.sum()
    rows = grouped.size()
    non_events = rows - events

    # chi-square consumes raw counts, not rates.
    table = pd.DataFrame({"no": non_events, "yes": events})
    chi2, p, dof, expected = chi2_contingency(table)
    valid = bool(expected.min() >= 5)

    rates = events / rows
    high, low = float(rates.max()), float(rates.min())
    gap = round(high - low, 4)
    ratio = round(high / low, 2) if low > 0 else None

    thin = [str(g) for g, n in rows.items() if n < SMALL_GROUP]
    two_groups = len(rates) == 2

    result = {
        "metric": metric,
        "group_by": group_by,
        "table": frame,
        "filters": applied or None,
        "rates": {str(k): round(float(v), 4) for k, v in rates.items()},
        "n_per_group": {str(k): int(v) for k, v in rows.items()},
        "events_per_group": {str(k): int(v) for k, v in events.items()},
        "n_total": int(len(dataframe)),
        # "gap" is a difference between two named groups; "spread" is
        # highest minus lowest across several, which is not a comparison of
        # any particular pair.
        "gap" if two_groups else "spread": gap,
        "ratio" if two_groups else "spread_ratio": ratio,
        # Not rounded: the strongest results in this project run to e-122,
        # and round(..., 6) would report every one of them as exactly 0.0.
        "p_value": float(p) if valid else None,
        "p_value_valid": valid,
        "p_value_note": None if valid else (
            "chi-square unreliable: an expected cell count is below 5, "
            "which the test's approximation requires. The rates and counts "
            "are still correct."
        ),
        "small_groups": thin or None,
    }

    # Comparing two disbursement months is a VINTAGE comparison, and the
    # maturity warning applies to it exactly as it does to the grouped case.
    if group_by in _CHRONOLOGICAL:
        result["cohort_note"] = (
            f"These groups are loans given out in different months, so this "
            f"compares cohorts. A cohort disbursed later has had less time to "
            f"go bad, which can look like a lower rate without being one."
        )

    return result


def crosstab_rate(metric, row_by, col_by, filters=None):
    """Rate for every combination of two grouping columns.

    Answers the confounder question: is the effect really about this
    variable, or about something riding along with it? Balochistan defaults
    more — but is that just because Balochistan takes more merchant
    advances? Reading across the Balochistan row answers it: if the rate is
    elevated in every product, the region effect is real.

    Since _prepare can bring a customer trait onto the loans frame, one axis
    may describe the PERSON and the other the LOAN — "default rate by savings
    status and loan purpose" is now a single call. And since it can derive
    disbursed_month, an axis may be WHEN — "default rate by month and
    product" is a vintage grid.

    `row_by` labels the rows, `col_by` the columns. The order changes the
    layout, not the numbers — pick whichever direction the comparison reads
    along.

    Returns a rate per cell, the row count behind each, and margins for both
    axes. Margins are computed by summing events and rows separately and
    dividing once, never by averaging the cell rates — a large group must
    count more than a small one.

    NO EVENT COUNTS COME BACK, DELIBERATELY. A cell has a rate and a row
    count; it does not have a "how many defaulted" figure. The narrator is
    told this explicitly, because inventing the missing half of an "X out of
    Y" sentence is exactly what it did before it was.

    Individual cells are thin by nature. The evidence in a grid is the
    consistency across cells, not any single figure; cells below SMALL_CELL
    rows are named in the return.
    """
    if row_by == col_by:
        raise ToolError(
            f"'{row_by}' cannot be crossed with itself. "
            f"For a single column, use aggregate_metric or compare_groups."
        )

    dataframe, frame, applied = _prepare(
        metric, (row_by, col_by), filters=filters
    )

    checked = _as_numeric(dataframe[metric])

    if not _is_binary(checked):
        raise ToolError(
            f"crosstab_rate shows a RATE per cell, so it needs a yes/no "
            f"outcome to count. '{metric}' is a measurement rather than an "
            f"outcome. Outcomes available: {list(_OUTCOME_METRICS)}."
        )

    grouped = checked.groupby(
        [dataframe[row_by], dataframe[col_by]], observed=True
    )
    events = grouped.sum().unstack()
    rows = grouped.size().unstack()

    rates = events / rows

    # Chronological axes sort by date, not by label order pandas happened to
    # produce. Same reasoning as aggregate_metric: a month axis out of date
    # order is unreadable.
    if row_by in _CHRONOLOGICAL:
        rates = rates.sort_index()
        rows = rows.sort_index()
    if col_by in _CHRONOLOGICAL:
        rates = rates.sort_index(axis=1)
        rows = rows.sort_index(axis=1)

    margins_rows = events.sum(axis=1) / rows.sum(axis=1)
    margins_columns = events.sum(axis=0) / rows.sum(axis=0)

    thin = [
        f"{r} / {c}"
        for (r, c), n in rows.stack().items()
        if n < SMALL_CELL
    ]

    result = {
        "metric": metric,
        "row_by": row_by,
        "col_by": col_by,
        "table": frame,
        "filters": applied or None,
        "rates": {
            str(r): {str(c): round(float(v), 4) for c, v in row.items()}
            for r, row in rates.to_dict("index").items()
        },
        "n_per_cell": {
            str(r): {str(c): int(v) for c, v in row.items()}
            for r, row in rows.to_dict("index").items()
        },
        # Summed, not averaged — a region with many loans must count more
        # than one with few.
        "row_margins": {
            str(k): round(float(v), 4) for k, v in margins_rows.items()
        },
        "col_margins": {
            str(k): round(float(v), 4) for k, v in margins_columns.items()
        },
        "n_total": int(len(dataframe)),
        "small_cells": thin or None,
        # Stated rather than left to be noticed. n_per_cell is a cell TOTAL,
        # and a narrator that reads it as an event count will invent a
        # denominator to pair with it — which is precisely what happened.
        "counts_note": (
            "n_per_cell is the TOTAL number of rows in each cell, not the "
            "number of events. This tool returns no event counts, so no "
            "'X out of Y' figure is available for any cell."
        ),
    }

    if row_by in _CHRONOLOGICAL or col_by in _CHRONOLOGICAL:
        result["cohort_note"] = (
            "One axis is the month the loan was given out, so those groups "
            "are cohorts rather than points in a trend. Later cohorts have "
            "had less time to go bad."
        )

    return result


def band_distribution(group_by, filters=None):
    """How the book splits across the levels of one column.

    Counts and shares, no outcome involved — "42% of customers sit in the
    bottom credit band". Pairs with the rate tools: those say how risky each
    band is, this says how much of the book sits there.

    THIS IS THE TOOL FOR "HOW MANY LOANS WERE GIVEN OUT EACH MONTH".
    group_by='disbursed_month' counts rows per month, which is lending
    VOLUME over time — a genuine trend, and one that carries no maturity
    caveat at all, because a count of disbursements is complete the moment
    they happen. Nothing has to wait to find out whether it counts.

    No metric, because nothing is being measured — rows are counted. The
    table is therefore chosen from the request itself, and THE POPULATION
    CHANGES with it — every customer, or only those who borrowed — which is
    why the return names its unit rather than leaving it implied.

    No small-group flag here, unlike the rate tools. They flag because a
    RATE on few rows is unstable: 60 defaults out of 304 could easily have
    been 50 or 70. A count carries no such uncertainty — 309 nano loans in
    Islamabad is simply how many there are, and its share is exact. Levels
    that are genuinely tiny show up in the counts themselves.
    """
    dataframe, frame, applied = _prepare(
        group_cols=(group_by,), filters=filters
    )

    counts = dataframe[group_by].value_counts()
    shares = dataframe[group_by].value_counts(normalize=True)

    # value_counts orders by frequency, which for a month column produces a
    # list nobody can read as a time series. Date order instead.
    chronological = group_by in _CHRONOLOGICAL
    if chronological:
        counts = counts.sort_index()
        shares = shares.sort_index()

    result = {
        "group_by": group_by,
        "table": frame,
        # The population changes with the table, so name it rather than
        # leaving the narrator to infer it from the table name.
        "unit": "loans" if frame == "default" else "customers",
        "filters": applied or None,
        "counts": {str(k): int(v) for k, v in counts.items()},
        "shares": {str(k): round(float(v), 4) for k, v in shares.items()},
        "n_total": int(len(dataframe)),
    }

    if chronological:
        result["ordering"] = "chronological"
        # No maturity caveat here, and the absence is deliberate. A count of
        # loans disbursed in a month is final the day the month ends. It is
        # the only genuine time series this dataset supports.
        result["trend_note"] = (
            "This is a count of loans given out in each month — lending "
            "volume over time. Unlike an outcome rate, it needs no maturity "
            "caveat: a disbursement is complete when it happens."
        )

    return result 