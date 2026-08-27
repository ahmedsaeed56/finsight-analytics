"""
Customer segmentation — fit K-Means, profile, validate, serialise.

    python -m src.models.train_segments

WHY THIS IS DIFFERENT
---------------------
No target, no train/test split in the usual sense, no AUC. K-Means finds groups
by distance; success means the clusters are distinct and nameable, not that they
match an answer.

FEATURE CHOICE HAS NO REFEREE
-----------------------------
Everywhere else, dropping a feature could be judged by whether the score moved.
Here nothing scores it. So the features do not improve the segments — they
DEFINE what the segments mean. Behavioural features (money and engagement) give
clusters a product team can act on; adding region and age would give clusters
split by geography and life stage that nobody can use.

TWO FEATURES REMOVED, FOR DIFFERENT REASONS
-------------------------------------------
total_value: 0.966 with total_txns. In a regression that means unreadable
coefficients; in K-Means it is worse — two near-identical columns give
transaction volume DOUBLE WEIGHT in every distance calculation.

has_insurance: the first fit produced two clusters (5,924 and 2,308) that were
near-identical on every behavioural dimension and separated only by insurance
being exactly 0.00 and 1.00. K-Means uses binary features as walls — after
scaling the 0/1 gap is large and nothing sits between, so the algorithm gets a
free clean split that costs almost no inertia and takes it instead of finding
real structure. Removing it dropped silhouette 0.261 -> 0.247 and produced a
segmentation that can actually be named. That trade is deliberate.

K WAS NOT CHOSEN BY THE DIAGNOSTICS
-----------------------------------
Silhouette was flat across K=2..10 (0.24-0.30) and inertia declined smoothly
with no elbow. There is no natural number of clusters because there are no
natural clusters — customers sit on a continuum. K=4 was chosen for
comparability with the four known segments, and that should be stated wherever
these clusters are presented.
"""
from __future__ import annotations

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import (
    CUSTOMERS_TRAIN, MODELS_DIR, RANDOM_STATE,
    SEGMENT_FEATURES, SEGMENT_FEATURES_TEST,
)

CLUSTER_FEATURES = [
    "avg_monthly_inflow_pkr",
    "total_txns",
    "active_months",
    "savings_balance_pkr",
    "wallet_tenure_months",
]
K = 4


def choose_k(X_scaled, k_range=range(2, 11)) -> pd.DataFrame:
    """Elbow and silhouette across a range of K.

    inertia always falls as K rises (at K = n it is zero), so the minimum is
    meaningless — look for a BEND. silhouette asks whether each point sits
    closer to its own cluster than the nearest other: above 0.5 is well
    separated, below 0.25 is close to arbitrary.
    """
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(X_scaled)
        rows.append({
            "k": k,
            "inertia": round(km.inertia_, 1),
            "silhouette": round(silhouette_score(X_scaled, labels), 3),
        })
    return pd.DataFrame(rows)


def main() -> None:
    train = pd.read_parquet(SEGMENT_FEATURES)
    test = pd.read_parquet(SEGMENT_FEATURES_TEST)
    print(f"train {train.shape}   test {test.shape}")

    X = train[CLUSTER_FEATURES]

    # Not optional. K-Means measures straight-line distance, and
    # savings_balance_pkr runs to hundreds of thousands while active_months runs
    # 0-12. Unscaled, savings alone would decide every cluster.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\nChoosing K:")
    print(choose_k(X_scaled).to_string(index=False))

    km = KMeans(n_clusters=K, n_init=10, random_state=RANDOM_STATE)
    train["cluster"] = km.fit_predict(X_scaled)

    print(f"\nCluster sizes (K={K}):")
    print(train["cluster"].value_counts().sort_index().to_string())

    print("\nCluster profile — this is where the names come from:")
    print(train.groupby("cluster")[CLUSTER_FEATURES].mean().round(1).to_string())

    # ---- validation against the answer key ----
    # segment_true was dropped at the feature stage so it could not leak in.
    # It is joined back only now, after fitting, to ask whether K-Means
    # rediscovered the designed groups without ever seeing them.
    truth = pd.read_parquet(CUSTOMERS_TRAIN)[["customer_id", "segment_true"]]
    labelled = train.merge(truth, on="customer_id", how="left")
    print("\nClusters vs segment_true (never seen during fitting):")
    print(pd.crosstab(labelled["cluster"], labelled["segment_true"]).to_string())
    print("\nRead row-wise: a row concentrated in one column means that real "
          "segment was recovered. A row spread evenly is 'everyone else'.")

    # Test customers get cluster assignments from the SAME fitted objects —
    # transform, never fit_transform, or the two splits sit in different spaces.
    test["cluster"] = km.predict(scaler.transform(test[CLUSTER_FEATURES]))
    print(f"\nTest cluster sizes:")
    print(test["cluster"].value_counts().sort_index().to_string())

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / "segment_model.joblib"
    joblib.dump(
        {"scaler": scaler, "kmeans": km, "features": CLUSTER_FEATURES},
        path,
    )
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
