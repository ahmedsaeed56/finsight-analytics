"""
src/tools/inference.py
======================

Tier 2 — the model tools. Prediction and explanation for one subject.

Mostly a wrapper layer. src/models/scoring.py already does the arithmetic
(probability, band, drivers) and importance.py already reads the coefficients;
what did not exist was the step in front of them — turn an id the router
supplied into the feature row those functions need, and fail readably when
the id is wrong.

TIER 1 vs TIER 2
----------------
Tier 1 measures what HAPPENED: 14% of these loans defaulted. Tier 2 says what
a model EXPECTS: this loan has a 0.34 probability. The first is a fact about
the file; the second is a claim from a model fitted on other data. The
narrator must not blur them, which is why every return here names its model.
"""

import pandas as pd

from src.config import PANEL_MONTHS
from src.models.importance import get_feature_importance as _get_feature_importance
from src.models.scoring import (
    _default,
    _range_flags,
    score_churn,
    score_default,
    score_population_probabilities,
    score_segment,
)
from src.tools.dataset import _frame
from src.tools.errors import ToolError


def predict_default(loan_id):
    """Default risk for one loan, with the reasons behind it.

    Parameters
    ----------
    loan_id
        A loan in the CURRENT dataset. The row must already exist — this
        scores a loan on the books. For a loan that does not exist yet, see
        simulate_loan.

    Returns
    -------
    dict — loan_id, model, probability, band, the drivers that moved it, and
    flags. The id is echoed because score_default returns only the numbers,
    and a narrator handling several calls needs to know which loan each
    describes.

    Raises
    ------
    ToolError
        The id is not in the current dataset.
    """
    df = _frame("default")

    # A one-row FRAME, not a Series. score_default selects columns from what
    # it is given, so .iloc[0] would hand it a Series and break the lookup.
    loan_row = df[df["loan_id"] == loan_id]

    # An unmatched filter returns zero rows rather than raising, so the empty
    # case has to be caught here — otherwise score_default fails on an empty
    # frame with something the router cannot act on.
    #
    # The count separates two different problems: 8,000 loans means the file
    # is fine and the id is wrong, so a retry can work; 0 would mean the wrong
    # file is loaded, and no retry will help. The format lets the router spot
    # that it sent a customer_id where a loan_id belongs.
    if loan_row.empty:
        raise ToolError(
            f"loan_id '{loan_id}' is not in the current dataset. "
            f"It holds {len(df):,} loans, with ids in the form L500000. "
            f"Check the id, or confirm the right file is loaded."
        )

    decision = score_default(loan_row)

    # Id first: it says what the whole result is about, and a dict built this
    # way reads in the order a narrator needs it.
    return {"loan_id": loan_id, **decision}


def predict_churn(customer_id):
    """Churn risk for one customer, with the reasons behind it.

    Same shape as predict_default, one table down: churn is a property of a
    CUSTOMER, so the id and the table both change.

    Parameters
    ----------
    customer_id
        A customer in the CURRENT dataset.

    Returns
    -------
    dict — customer_id, model, probability, band, drivers, flags.

    Raises
    ------
    ToolError
        The id is not in the current dataset.
    """
    df = _frame("churn")

    # A one-row FRAME, not a Series — score_churn selects columns from it.
    customer_row = df[df["customer_id"] == customer_id]

    # The count separates a wrong id from a wrong file; the format lets the
    # router spot that it sent a loan_id where a customer_id belongs.
    if customer_row.empty:
        raise ToolError(
            f"customer_id '{customer_id}' is not in the current dataset. "
            f"It holds {len(df):,} customers, with ids in the form C100000. "
            f"Check the id, or confirm the right file is loaded."
        )

    decision = score_churn(customer_row)

    return {"customer_id": customer_id, **decision}


def get_segment_profile(customer_id):
    """Which behavioural cluster one customer belongs to.

    Same lookup as predict_churn — segments are a property of a customer —
    but a different kind of answer. The other two tools return a risk
    probability; this returns a GROUP, with no probability and no band. See
    score_segment for why the shape differs.

    Reads the segment table rather than churn: it carries the full-panel
    behaviour the clustering was fitted on, and it holds every customer.

    Parameters
    ----------
    customer_id
        A customer in the CURRENT dataset.

    Returns
    -------
    dict — customer_id, model, cluster, distance_to_centre, margin,
    features_used, and the caveat that a cluster is not a risk level.

    Raises
    ------
    ToolError
        The id is not in the current dataset.
    """
    df = _frame("segment")

    customer_row = df[df["customer_id"] == customer_id]

    if customer_row.empty:
        raise ToolError(
            f"customer_id '{customer_id}' is not in the current dataset. "
            f"It holds {len(df):,} customers, with ids in the form C100000. "
            f"Check the id, or confirm the right file is loaded."
        )

    profile = score_segment(customer_row)

    return {"customer_id": customer_id, **profile}


def get_feature_importance(model, n=10):
    """What each model relies on across the whole population.

    Distinct from the `drivers` in predict_default and predict_churn: those
    explain ONE prediction, this describes the MODEL. Both read the same
    coefficients, so a narrator that confuses them will report a global
    pattern as a fact about one customer.

    Needs no dataset. Coefficients are a property of the fitted model, so
    this is the only Tier 2 tool that works before an upload.

    Parameters
    ----------
    model
        "default" or "churn". K-Means has no coefficients — for what defines
        the segments, see the features_used key in get_segment_profile.
    n
        How many numeric features to return. Categoricals come back in full,
        since each one is only meaningful beside its reference level.

    Returns
    -------
    dict — numeric and categorical features listed separately, the dropped
    reference level per categorical column, and the caveat explaining why a
    one-hot coefficient is a comparison rather than an absolute effect.

    Raises
    ------
    ToolError
        Unknown model name.
    """
    try:
        return _get_feature_importance(model, n=n)
    except ValueError as exc:
        # importance.py raises ValueError, which escapes the router
        # unhandled. Every other tool in this layer raises ToolError, which
        # the router is built to read and retry from — so the one job of
        # this wrapper is translating that.
        raise ToolError(str(exc)) from exc


# What each model name implies for the tool layer: which table holds the rows,
# which column identifies them, and what a row IS. Kept together because they
# vary as one — the default model scores loans, the churn model scores
# customers, and mixing those is how a return says "50 customers" about a
# ranking of loans.
#
# Deliberately separate from scoring.py's _POPULATION_MODELS, which holds the
# artifact and the band cuts. That file has no business knowing table names;
# this one has no business knowing cut points.
_POPULATION_TABLES = {
    "default": {"table": "default", "id_column": "loan_id", "unit": "loans"},
    "churn": {"table": "churn", "id_column": "customer_id", "unit": "customers"},
}


def score_population(model, limit=50):
    """Rank the whole book by risk and return the top of it.

    The tool the other four could not replace. predict_default answers "what
    about this loan" and needs an id, which the user would already have to
    know; aggregate_metric answers "what share defaulted", which is a rate
    with no names in it. Neither gives a retention team a list to call.

    RANKS, DOES NOT EXPLAIN. Per-row drivers are left out on purpose: the
    single-subject tools already return them, so the two compose — this says
    WHO, then predict_default(that_id) says WHY. Returning fifty driver lists
    would also bury the pattern the answer is actually about.

    FLAGS ARE CHECKED ON THE TOP ROWS ONLY. _range_flags costs a lookup per
    row, and the answer is `limit` rows, not the whole table — so scoring
    happens on everything and flagging on the fifty that survive the cut.

    TWO KINDS OF FLAG, TWO OUTCOMES. An out-of-range value leaves the row
    ranked with a caveat: the probability is real, just extrapolated. An
    UNSEEN CATEGORY moves the row to `not_scored`, because handle_unknown is
    "ignore" and an unseen region becomes all-zero columns — indistinguishable
    from AJK-GB, the dropped reference level and the lowest-default region.
    That number is not uncertain, it is wrong, and ranking it would place a
    Gilgit loan among the safest in the book.

    Parameters
    ----------
    model
        "default" ranks loans, "churn" ranks customers. The model decides the
        table, the id column, and the unit — they are not free choices.
    limit
        How many rows to return, highest probability first.

    Returns
    -------
    dict — model, unit, n_scored, limit, `ranked`, and `not_scored`.

    Raises
    ------
    ToolError
        Unknown model name.
    """
    if model not in _POPULATION_TABLES:
        raise ToolError(
            f"cannot rank by '{model}'. Available: "
            f"{sorted(_POPULATION_TABLES)}. Segments are behavioural groups "
            f"with no risk ordering, so there is nothing to rank them by."
        )

    spec = _POPULATION_TABLES[model]
    df = _frame(spec["table"])

    try:
        probabilities, bands = score_population_probabilities(df, model)
    except ValueError as exc:
        # scoring.py is a model module and cannot import from src.tools, so
        # it raises ValueError. Translating it here is the same job the
        # get_feature_importance wrapper does.
        raise ToolError(str(exc)) from exc

    # Built BEFORE sorting. probabilities and bands are positional arrays with
    # no index — they line up with df by row order only, so pairing them with
    # ids has to happen while that order still holds. Sorting first would pair
    # one loan's id with another loan's probability, the same trap the
    # counts.reindex line guards against in aggregate_metric.
    #
    # The frame's own index is kept, so a ranked row can be traced back to the
    # feature row the flag check needs.
    ranked = pd.DataFrame(
        {
            "id": df[spec["id_column"]].to_numpy(),
            "probability": probabilities,
            "band": bands,
        },
        index=df.index,
    )

    ranked = ranked.sort_values("probability", ascending=False).head(limit)

    rows = []
    not_scored = []

    for position, row in ranked.iterrows():
        flags = _range_flags(df.loc[[position]], spec["table"], _model_pipeline(model))

        unseen = [f for f in flags if f["kind"] == "unseen_category"]
        if unseen:
            not_scored.append({
                "id": row["id"],
                "reason": "; ".join(f["reason"] for f in unseen),
            })
            continue

        entry = {
            "id": row["id"],
            "probability": round(float(row["probability"]), 4),
            "band": str(row["band"]),
        }
        if flags:
            entry["flags"] = flags
        rows.append(entry)

    return {
        "model": model,
        # Named rather than implied: 8,000 loans and 15,000 customers are
        # different populations, and the narrator should not have to infer
        # which from the model name.
        "unit": spec["unit"],
        "id_column": spec["id_column"],
        "n_scored": int(len(df)),
        "limit": limit,
        "ranked": rows,
        # Rows that could not be ranked, and why. Separate from the ranking
        # because there is no honest probability to rank them by — the same
        # shape as compare_groups withholding p_value rather than returning a
        # number the caller cannot trust.
        "not_scored": not_scored or None,
    }


def _model_pipeline(model):
    """The fitted pipeline behind a model name, for the flag check."""
    from src.models.scoring import _POPULATION_MODELS

    return _POPULATION_MODELS[model][0]["pipeline"]


def simulate_loan(customer_id, amount_pkr, term_months, purpose):
    """Score a loan that does not exist yet.

    THE QUESTION THE OTHER TOOLS CANNOT ANSWER. predict_default explains a
    loan already on the books. This answers "should we lend to this person",
    which is what a lender actually asks before the loan exists.

    A HYBRID, NOT A PAYLOAD. This company lends to its own wallet users, so
    the applicant already has a customer row and twelve months of history —
    what they lack is a LOAN row. So the wallet half is real and only the
    loan terms are proposed.

    WHY BOTH TABLES. Neither alone carries what the default model needs.
    The SEGMENT table drops age_missing and income_band_missing — they are
    pipeline artifacts rather than customer traits, so clustering must not
    see them — but the default model uses both as features. The CHURN table
    keeps them, and holds every customer on an upload because the builder
    runs with drop_unlabeled=False.

    Conversely, churn's total_counts and total_amount cover the FIRST SIX
    MONTHS only, while the default model's wallet averages are per-month over
    the whole pre-loan window. Taking them from churn would silently halve
    every activity figure. So the profile comes from churn and the full-panel
    wallet totals come from segment.

    WHY THE WINDOW IS PANEL_MONTHS - 1
    ----------------------------------
    A loan disbursed today would look back over the entire panel, so twelve
    months is the literal answer. But every loan in training was disbursed
    INSIDE the panel, so months_available ran 1 to 11 and never reached 12.
    Passing 12 puts every simulation outside the trained range, and a flag
    that fires on every call forever tells a reader nothing — it only teaches
    them to ignore the list.

    Eleven is the honest analogue: a loan disbursed at the very end of the
    panel, which is the closest thing to "today" the model has ever seen.

    THE LIMITATION TO STATE WHENEVER THIS IS USED. The model only ever saw
    loans that were DISBURSED. It never saw an application someone declined,
    so it ranks the kind of applicant this company already lends to and says
    nothing about the ones the current process turns away. That is the
    reject-inference problem, and it is inherent to real credit data too.

    Parameters
    ----------
    customer_id
        An existing customer. Their wallet behaviour and profile are read
        from the current dataset.
    amount_pkr, term_months, purpose
        The proposed loan.

    Returns
    -------
    dict — the inputs echoed, plus probability, band, drivers, flags, and the
    reject-inference caveat.

    Raises
    ------
    ToolError
        The customer is not in the dataset, or the loan terms are invalid.
    """
    churn_df = _frame("churn")
    segment_df = _frame("segment")

    profile = churn_df[churn_df["customer_id"] == customer_id]
    wallet = segment_df[segment_df["customer_id"] == customer_id]

    if profile.empty or wallet.empty:
        raise ToolError(
            f"customer_id '{customer_id}' is not in the current dataset. "
            f"It holds {len(segment_df):,} customers, with ids in the form "
            f"C100000. Check the id, or confirm the right file is loaded."
        )

    if amount_pkr <= 0:
        raise ToolError(f"amount_pkr must be positive, not {amount_pkr}.")

    wallet_row = wallet.iloc[0]
    inflow = float(profile["avg_monthly_inflow_pkr"].iloc[0])

    if inflow <= 0:
        raise ToolError(
            f"customer '{customer_id}' has no recorded monthly inflow, so "
            f"the loan-to-income ratio — the model's strongest driver — "
            f"cannot be computed."
        )

    # Start from the churn row: it carries the profile columns AND the two
    # missingness flags the default model needs.
    proposed = profile.copy()

    proposed["amount_pkr"] = float(amount_pkr)
    proposed["term_months"] = int(term_months)
    proposed["purpose"] = purpose

    # The H1 driver, and the whole reason this tool is useful: the proposed
    # amount measured against income the customer already has.
    proposed["inflow_to_loan_ratio"] = float(amount_pkr) / inflow

    # One window, used four times. See the docstring for why it is
    # PANEL_MONTHS - 1 rather than the panel length itself.
    window = PANEL_MONTHS - 1

    # Full-panel wallet figures, from segment, divided by that window.
    proposed["months_available"] = window
    proposed["average_txns_per_mon"] = float(wallet_row["total_txns"]) / window
    proposed["average_value_per_mon"] = float(wallet_row["total_value"]) / window
    proposed["active_ratio"] = float(wallet_row["active_months"]) / window

    missing = [c for c in _default["features"] if c not in proposed.columns]
    if missing:
        raise ToolError(
            f"cannot simulate: the current dataset is missing "
            f"{sorted(missing)}, which the default model requires."
        )

    decision = score_default(proposed)

    return {
        "customer_id": customer_id,
        "proposed": {
            "amount_pkr": float(amount_pkr),
            "term_months": int(term_months),
            "purpose": purpose,
            "inflow_to_loan_ratio": round(
                float(proposed["inflow_to_loan_ratio"].iloc[0]), 4
            ),
        },
        **decision,
        "caveat": (
            "The model was fitted only on loans that were disbursed — it "
            "never saw an application that was declined. It can rank this "
            "applicant against people the company already lends to, but says "
            "nothing about applicants the current process turns away."
        ),
    } 