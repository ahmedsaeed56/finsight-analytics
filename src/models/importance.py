"""
Global feature importance — what each model relies on across the population.

Distinct from ``top_drivers`` in scoring.py: that explains ONE prediction, this
describes the model. Both come from the same coefficients, but they answer
different questions and the tool layer must not confuse them.

THE REFERENCE-LEVEL PROBLEM
---------------------------
OneHotEncoder(drop="first") drops the alphabetically first category, so every
surviving region coefficient means "relative to AJK-GB" — the smallest region
(~337 customers) and the lowest-default group. That inflates all of them:
Balochistan comes out at +0.907, ahead of inflow_to_loan_ratio at +0.498, which
would make it look like the top driver. It is not. It is an artifact of an
unusually clean baseline.

Returning the coefficients bare would have the narration layer report that as
fact. So categorical features are returned in a SEPARATE list with the
reference level named, and the caveat travels with the data rather than living
only in a comment.
"""
from __future__ import annotations

import joblib
import numpy as np

from src.config import CHURN_MODEL, DEFAULT_MODEL, FEATURE_LABELS

_MODELS = {
    "default": joblib.load(DEFAULT_MODEL),
    "churn": joblib.load(CHURN_MODEL),
}


def _split_name(raw: str) -> tuple[str, str | None]:
    """``nom__region_Balochistan`` -> ("region", "Balochistan").

    Returns (base column, level) — level is None for non-categorical features.
    """
    body = raw.split("__", 1)[-1]
    if not raw.startswith("nom__"):
        return body, None
    for base in FEATURE_LABELS:
        if body.startswith(base + "_"):
            return base, body[len(base) + 1:]
    return body, None


def get_feature_importance(model: str, n: int = 10) -> dict:
    """Rank features by absolute coefficient.

    Because the pipeline scales first, magnitudes are comparable across
    features measured on different units — a coefficient on credit_score
    (300-850) and one on dependents (0-8) mean the same thing.

    Returns numeric and categorical features separately. Categoricals carry
    their reference level and a caveat, because a one-hot coefficient is a
    comparison against a baseline rather than an absolute effect.
    """
    if model not in _MODELS:
        raise ValueError(f"unknown model '{model}' — expected one of {list(_MODELS)}")

    pipeline = _MODELS[model]["pipeline"]
    prep = pipeline.named_steps["prep"]
    coefs = pipeline.named_steps["model"].coef_[0]
    names = prep.get_feature_names_out()

    numeric, categorical = [], []
    seen_bases: dict[str, list[str]] = {}

    for raw, coef in zip(names, coefs):
        base, level = _split_name(raw)
        entry = {
            "feature": FEATURE_LABELS.get(base, base.replace("_", " ")),
            "column": base,
            "coefficient": round(float(coef), 4),
            "direction": "increases risk" if coef > 0 else "decreases risk",
        }
        if level is None:
            numeric.append(entry)
        else:
            entry["level"] = level
            categorical.append(entry)
            seen_bases.setdefault(base, []).append(level)

    numeric.sort(key=lambda d: abs(d["coefficient"]), reverse=True)
    categorical.sort(key=lambda d: abs(d["coefficient"]), reverse=True)

    # Identify the dropped reference for each categorical column. It is the one
    # allowed value that produced no coefficient.
    references = {}
    for base, levels in seen_bases.items():
        idx = prep.transformers_[1][2].index(base) if base in prep.transformers_[1][2] else None
        if idx is not None:
            all_levels = list(prep.named_transformers_["nom"].categories_[idx])
            missing = [v for v in all_levels if v not in levels]
            references[base] = missing[0] if missing else None

    for entry in categorical:
        entry["compared_to"] = references.get(entry["column"])

    return {
        "model": model,
        "numeric_features": numeric[:n],
        "categorical_features": categorical,
        "reference_levels": references,
        "caveat": (
            "Categorical coefficients are differences from a reference level, "
            "not absolute effects. For region the reference is "
            f"{references.get('region')}, which is the smallest group and has "
            "the lowest outcome rate — so every other region's coefficient is "
            "inflated. Compare regions to each other, not to zero."
        ),
        "interpretation": (
            "Coefficients are on scaled features, so magnitudes are comparable "
            "across columns with different units. Sign gives direction. These "
            "are associations, not causes."
        ),
    }
