"""
Default risk model — train, score once on test, serialise.

    python -m src.models.train_default

MODEL CHOICE
------------
Logistic regression, chosen over tuned XGBoost (val AUC 0.758 vs 0.773) because
the 0.015 gap sits on 181 validation defaults — inside noise — while the
coefficients stay readable and the model stays explainable, which regulated
lending requires in practice.

The stronger reason is what the tuner did: given a grid reaching max_depth 8, it
selected max_depth=1. Every tree becomes a single split on a single feature, so
the model has NO interaction terms at all — structurally the same family as
logistic regression. An algorithm free to find any pattern chose the simplest
possible structure, which says the signal here is essentially additive.

DEPLOYMENT NOTE
---------------
The pipeline serialised here handles encoding and scaling only. It does NOT
build features — that happens upstream in src.features.default, which any
scoring service must call first.
"""
from __future__ import annotations

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, classification_report, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder, StandardScaler,
)

from src.config import (
    DEFAULT_FEATURES, DEFAULT_FEATURES_TEST, MODELS_DIR, RANDOM_STATE,
)
from src.features.bands import BAND_COLUMNS

TARGET = "defaulted"
ID_COLS = ["loan_id", "customer_id", "disbursed_date"]

# Removed after a correlation check: total_txns/total_value came back at 0.963,
# and the model split one effect across them into offsetting weights (+0.50 and
# -0.35 on the same underlying behaviour). Dropping all four moved AUC from
# 0.75755 to 0.75765 — no predictive cost, and the coefficients became readable.
COLLINEAR = ["total_txns", "total_value", "average_value_per_mon", "active_months"]

NOMINAL = ["region", "purpose"]
ORDINAL = ["declared_income_band"]
INCOME_ORDER = [["<25k", "25-50k", "50-100k", "100-250k", "250k+"]]


def make_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=ID_COLS + [TARGET] + COLLINEAR +BAND_COLUMNS, errors="ignore")
    y = df[TARGET].astype(int)
    return X, y


def build_pipeline(numeric: list[str]) -> Pipeline:
    """Encoding + scaling + model as one object.

    Bundled deliberately: inside cross-validation the scaler is refit on each
    fold's training portion only. Scaling the whole frame first and then
    cross-validating would let every fold's scaler see its own validation rows —
    a subtler leak than the ones in the feature stage, but the same species.

    handle_unknown='ignore' is why this beats get_dummies: a category present in
    test but absent from train produces zeros rather than a shape mismatch.
    """
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("nom", OneHotEncoder(drop="first", handle_unknown="ignore"), NOMINAL),
            ("ord", OrdinalEncoder(categories=INCOME_ORDER), ORDINAL),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("prep", preprocess),
        ("model", LogisticRegression(
            class_weight="balanced",   # 14.1% positives; without this the model
                                       # learns to always predict "no default"
            max_iter=1000,
            random_state=RANDOM_STATE,
        )),
    ])


def report(name: str, y_true, y_pred, y_proba) -> None:
    print(f"\n--- {name} ---")
    print(classification_report(y_true, y_pred, digits=3))
    print(f"ROC-AUC          {roc_auc_score(y_true, y_proba):.4f}")
    print(f"Avg precision    {average_precision_score(y_true, y_proba):.4f}")
    print(f"Base rate        {y_true.mean():.4f}")


def threshold_table(y_true, y_proba, amounts: pd.Series) -> pd.DataFrame:
    """Operating points against the A/B test's 15% volume guardrail.

    A risk score does not decide anything; someone picks a cutoff. The business
    constraint established by the A/B test is that book-level disbursed volume
    must not fall more than 15%, so the column that matters is volume retained,
    not F1 — F1 weights precision and recall equally, which has nothing to do
    with lending economics.
    """
    rows = []
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        flagged = (y_proba >= t).astype(int)
        approved = flagged == 0
        rows.append({
            "threshold": t,
            "defaults_caught": (flagged[y_true == 1] == 1).mean(),
            "precision": (y_true[flagged == 1] == 1).mean() if flagged.sum() else float("nan"),
            "volume_retained": amounts[approved].sum() / amounts.sum(),
        })
    return pd.DataFrame(rows).round(3)


def main() -> None:
    train = pd.read_parquet(DEFAULT_FEATURES)
    test = pd.read_parquet(DEFAULT_FEATURES_TEST)
    print(f"train {train.shape}   test {test.shape}")

    X, y = make_xy(train)
    X_test, y_test = make_xy(test)

    # Column order must match between splits or the ColumnTransformer silently
    # scales the wrong columns.
    X_test = X_test[X.columns]

    numeric = X.select_dtypes(exclude="category").columns.tolist()
    assert len(numeric) + len(NOMINAL) + len(ORDINAL) == X.shape[1], \
        "column groups do not cover X exactly once"

    # stratify keeps the 14% rate in both halves; without it the split itself
    # would move the scores.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipe = build_pipeline(numeric)
    pipe.fit(X_tr, y_tr)
    report("VALIDATION", y_val, pipe.predict(X_val), pipe.predict_proba(X_val)[:, 1])

    weights = pd.Series(
        pipe.named_steps["model"].coef_[0],
        index=pipe.named_steps["prep"].get_feature_names_out(),
    ).sort_values(key=abs, ascending=False)
    print("\nTop coefficients (sign = direction, magnitude = influence):")
    print(weights.head(8).round(3).to_string())

    # ---- test: scored ONCE, after every modelling decision is locked ----
    test_proba = pipe.predict_proba(X_test)[:, 1]
    report("TEST (held out, scored once)", y_test, pipe.predict(X_test), test_proba)

    print("\nOperating points vs the 15% volume guardrail:")
    print(threshold_table(y_test, test_proba, test["amount_pkr"]).to_string(index=False))

    # ---- refit on everything, then serialise ----
    # More data makes a better model, and the honest test estimate is already
    # recorded above.
    final = build_pipeline(numeric)
    final.fit(pd.concat([X, X_test]), pd.concat([y, y_test]))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / "default_model.joblib"
    joblib.dump({"pipeline": final, "features": list(X.columns)}, path)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
