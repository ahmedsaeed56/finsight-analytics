"""
Scoring layer — probability, band, and drivers for a single subject.

This is what the LLM tools call. Everything here is deterministic arithmetic on
a fitted model; the language layer receives these facts and turns them into a
sentence. It never computes and never estimates.

WHY NOT SHAP
------------
Both shipped models are logistic regression, where a feature's contribution to
one prediction is exactly ``coefficient x scaled_value``. SHAP's LinearExplainer
computes the same quantity (up to a constant per-feature offset that shifts
every contribution equally and so cannot change the ranking). The dependency
would buy nothing here. It IS the right tool for a tree model, where there is no
coefficient to read — noted in the README as the path if the model family
changes.

EVERY PREDICTION CARRIES ITS OWN CAVEAT
---------------------------------------
A score for a customer the model has never seen the like of looks exactly like
a score for one it understands. `flags` is what separates them — see
_range_flags.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from src.config import (
    BAND_LABELS, BASELINE, CHURN_BAND_CUTS, CHURN_MODEL,
    DEFAULT_BAND_CUTS, DEFAULT_MODEL, FEATURE_LABELS, SEGMENT_MODEL,
    TOP_DRIVERS_N,
)

# Loaded once at import. joblib.load reads from disk and rebuilds the pipeline,
# which is slow enough to dominate latency if repeated per request.
_default = joblib.load(DEFAULT_MODEL)
_churn = joblib.load(CHURN_MODEL)
_segment = joblib.load(SEGMENT_MODEL)

# The training population's bounds. Loaded here beside the models because it is
# only valid FOR them — retrain and the distributions move too.
_baseline = json.loads(BASELINE.read_text(encoding="utf-8"))

# Model name -> the fitted artifact and its band cuts. Module level, because
# the two vary together: the churn model's probabilities mean nothing against
# the default model's cuts, and two separate lookups could drift apart.
_POPULATION_MODELS = {
    "default": (_default, DEFAULT_BAND_CUTS),
    "churn": (_churn, CHURN_BAND_CUTS),
}


def assign_band(probability: float, cuts: tuple[float, float]) -> str:
    """Map a probability onto low / medium / high.

    Cuts live in config because they are a policy choice, not a model output —
    a risk officer moving a boundary should not have to touch model code.
    """
    low, high = cuts
    if probability < low:
        return BAND_LABELS[0]
    if probability < high:
        return BAND_LABELS[1]
    return BAND_LABELS[2]


def _label(raw_name: str) -> tuple[str, str]:
    """Undo the ColumnTransformer's renaming.

    Names arrive as ``num__credit_score``, ``nom__region_Balochistan``,
    ``ord__declared_income_band``. Without this the narration layer would say
    "num__inflow_to_loan_ratio" to an end user.

    Returns (human label, original column name).
    """
    body = raw_name.split("__", 1)[-1]
    if body in FEATURE_LABELS:
        return FEATURE_LABELS[body], body
    # one-hot: "region_Balochistan" -> base "region", level "Balochistan"
    for base, label in FEATURE_LABELS.items():
        if body.startswith(base + "_"):
            return f"{label}: {body[len(base) + 1:]}", base
    return body.replace("_", " "), body


def top_drivers(pipeline, row: pd.DataFrame, n: int = TOP_DRIVERS_N) -> list[dict]:
    """The n features that moved this prediction most.

    contribution = coefficient x transformed_value. Sign gives direction,
    magnitude gives strength — and because the pipeline scales first, those
    magnitudes are comparable across features measured on different units.

    ``value`` is the subject's ORIGINAL figure (4.2), not the scaled one
    (1.87 SD), because the scaled number means nothing to a reader.
    """
    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]

    transformed = prep.transform(row)
    if hasattr(transformed, "toarray"):        # OneHotEncoder can return sparse
        transformed = transformed.toarray()

    contributions = model.coef_[0] * np.asarray(transformed).ravel()
    names = prep.get_feature_names_out()

    drivers = []
    for i in np.argsort(np.abs(contributions))[::-1][:n]:
        label, column = _label(names[i])
        value = None if names[i].startswith("nom__") else (
            row[column].iloc[0] if column in row.columns else None
        )
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        drivers.append({
            "feature": label,
            "column": column,
            "value": value,
            "direction": "increases risk" if contributions[i] > 0 else "decreases risk",
            "contribution": round(float(contributions[i]), 4),
        })
    return drivers


def _known_categories(pipeline) -> dict:
    """Every categorical level the fitted encoder has seen.

    Read off the ENCODER rather than from config, because what matters is what
    the model knows, not what the schema permits. Those can differ: a value
    added to VALID_REGIONS after training is legal in cleaning and still
    unscoreable.

    transformers_[1] is the "nom" entry, and [2] is its column list; the
    encoder's categories_ come back in that same order.
    """
    prep = pipeline.named_steps["prep"]
    columns = prep.transformers_[1][2]
    encoder = prep.named_transformers_["nom"]
    return {
        col: set(levels)
        for col, levels in zip(columns, encoder.categories_)
    }


def _range_flags(features_row: pd.DataFrame, table: str, pipeline) -> list[dict]:
    """Whether this row sits inside everything the model was trained on.

    TWO FAILURES, ONE MECHANISM.

    A numeric value outside the training range means the model EXTRAPOLATES.
    Logistic regression is a straight line in log-odds, so it extends forever
    without complaint, and the sigmoid squashes the result into a perfectly
    ordinary-looking probability. Nothing else in the output distinguishes a
    well-supported prediction from a guess — reference tenure starts at about
    nine months, so a brand-new customer gets a confident number built on
    nothing.

    A categorical value the encoder never saw is worse. handle_unknown is
    "ignore", so an unseen region becomes all-zero columns — indistinguishable
    from AJK-GB, which is the DROPPED reference level and the lowest-default
    region. A Gilgit loan would score as the safest region in the book.

    Takes the FULL feature row, not the model's column subset: a profile
    column the model does not use can still be out of range, and that is worth
    knowing.

    Returns a list, empty when the row is fully in range. Flags rather than
    withholds — the same principle as compare_groups returning valid rates
    beside a null p_value. One number is uncertain; the rest still stands.
    """
    flags = []

    for col, known in _known_categories(pipeline).items():
        if col not in features_row.columns:
            continue
        value = features_row[col].iloc[0]
        if pd.notna(value) and value not in known:
            flags.append({
                "column": col,
                "value": str(value),
                "kind": "unseen_category",
                "reason": (
                    f"'{value}' was not in the training data for {col}. The "
                    f"encoder cannot represent it, so this prediction is "
                    f"unreliable."
                ),
            })

    for col, entry in _baseline[table]["columns"].items():
        if entry["kind"] != "numeric" or col not in features_row.columns:
            continue
        value = features_row[col].iloc[0]
        if pd.isna(value):
            continue
        if value < entry["min"] or value > entry["max"]:
            flags.append({
                "column": col,
                "value": float(value),
                "kind": "out_of_range",
                "reason": (
                    f"{value} is outside the training range for {col} "
                    f"({entry['min']} to {entry['max']}). The prediction is "
                    f"extrapolated — its direction is probably right, its "
                    f"magnitude is unverified."
                ),
            })

    return flags


def score_default(features_row: pd.DataFrame) -> dict:
    """Score one loan application.

    ``features_row`` is one row from the default feature table, already built
    by src.features.default. This does NOT build features — the pipeline
    handles encoding and scaling only.
    """
    row = features_row[_default["features"]]
    prob = float(_default["pipeline"].predict_proba(row)[0, 1])
    return {
        "model": "default",
        "probability": round(prob, 4),
        "band": assign_band(prob, DEFAULT_BAND_CUTS),
        "drivers": top_drivers(_default["pipeline"], row),
        # Empty when the row sits inside everything the model saw. Non-empty
        # means the number came back anyway and should not be trusted at face
        # value.
        "flags": _range_flags(features_row, "default", _default["pipeline"]),
    }


def score_churn(features_row: pd.DataFrame) -> dict:
    """Score one customer for churn risk. Same contract as score_default."""
    row = features_row[_churn["features"]]
    prob = float(_churn["pipeline"].predict_proba(row)[0, 1])
    return {
        "model": "churn",
        "probability": round(prob, 4),
        "band": assign_band(prob, CHURN_BAND_CUTS),
        "drivers": top_drivers(_churn["pipeline"], row),
        "flags": _range_flags(features_row, "churn", _churn["pipeline"]),
    }


def score_segment(features_row: pd.DataFrame) -> dict:
    """Assign one customer to a behavioural cluster.

    A different KIND of answer from the other two, and the return shape says
    so. K-Means produces a GROUP, not a probability: there is no risk to
    band, and no coefficient to multiply into drivers. Returning a fabricated
    "probability" key for symmetry would let a narrator treat a cluster label
    as a risk score.

    HOW FAR, NOT JUST WHICH
    -----------------------
    `distance_to_centre` is the honest half of this answer. Silhouette was
    flat (~0.247) across K=2..10 with no elbow, which means customers sit on
    a CONTINUUM rather than in natural groups — every customer gets a label
    whether or not they resemble it. The distance says how well this one
    actually fits, and `margin` says how much closer their cluster is than
    the runner-up. A small margin means the label could easily have gone the
    other way.

    NO RANGE FLAGS. K-Means has no encoder and no extrapolation problem — an
    unusual customer simply lands far from every centroid, and
    distance_to_centre already says so.
    """
    row = features_row[_segment["features"]]

    # transform, never fit_transform — refitting here would put this customer
    # in a different space from every other one, and the assignment would be
    # meaningless.
    scaled = _segment["scaler"].transform(row)

    # predict returns an array even for one row; int() takes the value out and
    # converts numpy's int64 to a plain int, which JSON can serialise.
    cluster = int(_segment["kmeans"].predict(scaled)[0])

    # transform gives the distance to EVERY centroid, so the two nearest are
    # available without a second call.
    distances = _segment["kmeans"].transform(scaled)[0]
    nearest, runner_up = sorted(distances)[:2]

    return {
        "model": "segment",
        "cluster": cluster,
        # Scaled units (standard deviations), not rupees or months — the
        # distance is measured after scaling, so it has no natural unit.
        "distance_to_centre": round(float(nearest), 4),
        "margin": round(float(runner_up - nearest), 4),
        "features_used": list(_segment["features"]),
        "caveat": (
            "Clusters are behavioural groups, not risk levels — a higher "
            "cluster number is not worse. K=4 was chosen for comparability "
            "with the four designed segments, not found in the data: "
            "silhouette was flat across K=2..10 with no elbow, so customers "
            "sit on a continuum. A small margin means this customer sat "
            "almost equally close to another cluster."
        ),
    }


def band_column(probabilities, cuts: tuple[float, float]) -> pd.Categorical:
    """Vectorised band assignment for a whole population.

    For the aggregate tools ("how many applications land in high risk"), where
    calling assign_band per row would be needlessly slow.
    """
    return pd.cut(
        probabilities,
        bins=[0, cuts[0], cuts[1], 1.0],
        labels=list(BAND_LABELS),
        include_lowest=True,
    )


def score_population_probabilities(features, model):
    """Score a whole table at once.

    The batch counterpart to score_default and score_churn. Those answer
    "what about this one"; this answers "what about all of them", which is
    what a ranked list needs.

    One vectorised call, not a loop. predict_proba takes the entire frame and
    returns a probability per row — calling score_default 15,000 times would
    rebuild the same transform 15,000 times for the same answer.

    NO DRIVERS HERE. top_drivers costs a transform per row and the ranked list
    only needs to say WHO. The single-subject tools already explain WHY, so
    the two compose rather than duplicate.

    Returns
    -------
    (probabilities, bands) — positional arrays in the frame's own row order.
    The caller must keep that order when pairing them with ids.
    """
    if model not in _POPULATION_MODELS:
        raise ValueError(
            f"unknown model '{model}' — expected one of "
            f"{list(_POPULATION_MODELS)}"
        )

    artifact, cuts = _POPULATION_MODELS[model]

    # The fitted columns, in the order the pipeline expects. Same selection
    # score_default makes for one row.
    rows = features[artifact["features"]]

    # [:, 1] not [0, 1]: every row, positive-class column. The single-row
    # scorers take row zero because they only have one.
    probabilities = artifact["pipeline"].predict_proba(rows)[:, 1]

    # band_column was written for exactly this and has never been called —
    # assign_band per row would be needlessly slow on a whole table.
    bands = band_column(probabilities, cuts)

    return probabilities, bands 