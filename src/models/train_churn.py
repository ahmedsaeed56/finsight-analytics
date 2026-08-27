"""
Churn model — train, score once on test, serialise.

    python -m src.models.train_churn

MODEL CHOICE
------------
Logistic regression. Tuned XGBoost tied on ROC-AUC (0.710 vs 0.709) and lost on
average precision (0.164 vs 0.172), the metric that actually matters at a 7.7%
base rate. The tuner picked max_depth=4 here — it wanted feature interactions,
unlike the default model — but they did not survive out of sample: CV average
precision 0.191 against 0.164 on validation, which is overfitting to the folds.

METRIC CHOICE
-------------
Average precision (PR-AUC) is the primary metric, not ROC-AUC. With 7.7%
positives, ROC-AUC is flattered by how easy the large negative class is. Average
precision cares only about how well the rare class is ranked, and its floor is
the base rate itself (0.077) rather than 0.5.

HONEST CEILING
--------------
Average precision 0.172 against a 0.077 base rate is 2.2x lift. This is a
targeting aid — it tells a retention team where to look first — not a
prediction of who is leaving.
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
    CHURN_FEATURES, CHURN_FEATURES_TEST, MODELS_DIR, RANDOM_STATE,
)
from src.features.bands import BAND_COLUMNS

TARGET = "churned_12m"
ID_COLS = ["customer_id"]

# first/last/difference are redundant by construction — difference IS the other
# two subtracted. All three were kept in the parquet because the choice is
# per-model (trees tolerate it, linear models do not); difference is kept here
# because it is the only feature carrying DIRECTION rather than volume, and it
# correlated at ~0 with every volume feature.
# total_amount dropped: 0.943 with total_counts, near-duplicate.
REDUNDANT = ["first", "last", "total_amount"]

NOMINAL = ["region"]
ORDINAL = ["declared_income_band"]
INCOME_ORDER = [["<25k", "25-50k", "50-100k", "100-250k", "250k+"]]


def make_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # Y is the positive class: churn is what we are trying to detect, so recall
    # is measured against it.
    y = df[TARGET].map({"Y": 1, "N": 0})
    assert y.notna().all(), "unmapped label value — check for stray whitespace"
    X = df.drop(columns=ID_COLS + [TARGET] + REDUNDANT +BAND_COLUMNS, errors="ignore")
    return X, y.astype(int)


def build_pipeline(numeric: list[str]) -> Pipeline:
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
            class_weight="balanced",   # 7.7% positives — harder than default's 14%
            max_iter=1000,
            random_state=RANDOM_STATE,
        )),
    ])


def report(name: str, y_true, y_pred, y_proba) -> None:
    base = y_true.mean()
    ap = average_precision_score(y_true, y_proba)
    print(f"\n--- {name} ---")
    print(classification_report(y_true, y_pred, digits=3))
    print(f"ROC-AUC          {roc_auc_score(y_true, y_proba):.4f}")
    print(f"Avg precision    {ap:.4f}   (base rate {base:.4f}, lift {ap / base:.2f}x)")


def threshold_table(y_true, y_proba) -> pd.DataFrame:
    """Operating points for a retention campaign.

    Unlike lending there is no hard guardrail — contacting a customer costs an
    offer, not a lost loan. So this produces options, not a verdict. The right
    point depends on offer cost against customer lifetime value, which is a
    business input rather than a modelling one.
    """
    base = y_true.mean()
    rows = []
    for t in [0.4, 0.5, 0.6, 0.7, 0.8]:
        flagged = (y_proba >= t).astype(int)
        if flagged.sum() == 0:
            continue
        prec = (y_true[flagged == 1] == 1).mean()
        rows.append({
            "threshold": t,
            "share_contacted": flagged.mean(),
            "churners_caught": (flagged[y_true == 1] == 1).mean(),
            "precision": prec,
            "lift": prec / base,
        })
    return pd.DataFrame(rows).round(3)


def main() -> None:
    train = pd.read_parquet(CHURN_FEATURES)
    test = pd.read_parquet(CHURN_FEATURES_TEST)
    print(f"train {train.shape}   test {test.shape}")

    X, y = make_xy(train)
    X_test, y_test = make_xy(test)
    X_test = X_test[X.columns]

    numeric = X.select_dtypes(exclude="category").columns.tolist()
    assert len(numeric) + len(NOMINAL) + len(ORDINAL) == X.shape[1], \
        "column groups do not cover X exactly once"

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
    print("\nTop coefficients:")
    print(weights.head(8).round(3).to_string())

    # ---- test: scored ONCE ----
    test_proba = pipe.predict_proba(X_test)[:, 1]
    report("TEST (held out, scored once)", y_test, pipe.predict(X_test), test_proba)

    print("\nRetention campaign operating points:")
    print(threshold_table(y_test, test_proba).to_string(index=False))

    final = build_pipeline(numeric)
    final.fit(pd.concat([X, X_test]), pd.concat([y, y_test]))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / "churn_model.joblib"
    joblib.dump({"pipeline": final, "features": list(X.columns)}, path)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
