"""
Day 4 - Baseline unsupervised anomaly detection models.

Trains two unsupervised models on the label-free Day 3 feature table
(user_features.csv) and evaluates their rankings against the withheld
ground truth (labels.csv) - labels are used ONLY here, for evaluation,
never as a model input.

Models:
  1. Isolation Forest (from-scratch implementation, see
     src/models/isolation_forest.py)
  2. Mahalanobis-distance detector (stand-in for One-Class SVM, see
     src/models/mahalanobis.py for why)

Metrics (all implemented manually with NumPy, no sklearn.metrics):
  - Precision / Recall / F1 at a fixed "flag top-K" threshold
    (K = number of known malicious users, a common way to evaluate
    ranking-based detectors when the base rate is small)
  - ROC-AUC via the Mann-Whitney U / rank-sum formula
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from models.isolation_forest import IsolationForestScratch
from models.mahalanobis import MahalanobisDetector

DATA_DIR = "data/processed/full"


def standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std


def roc_auc(y_true, scores):
    """Manual ROC-AUC via the rank-sum (Mann-Whitney U) formula."""
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)

    # average ranks for ties
    sorted_scores = scores[order]
    sorted_ranks = ranks[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = sorted_ranks[i:j + 1].mean()
            sorted_ranks[i:j + 1] = avg
        i = j + 1
    ranks[order] = sorted_ranks

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc


def precision_recall_f1_at_k(y_true, scores, k):
    top_k_idx = np.argsort(-scores)[:k]
    predicted = np.zeros_like(y_true)
    predicted[top_k_idx] = 1
    tp = ((predicted == 1) & (y_true == 1)).sum()
    fp = ((predicted == 1) & (y_true == 0)).sum()
    fn = ((predicted == 0) & (y_true == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def roc_curve_points(y_true, scores, n_points=50):
    thresholds = np.linspace(scores.min(), scores.max(), n_points)[::-1]
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    tprs, fprs = [], []
    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        tprs.append(tp / n_pos if n_pos else 0)
        fprs.append(fp / n_neg if n_neg else 0)
    return np.array(fprs), np.array(tprs)


def main():
    features = pd.read_csv(f"{DATA_DIR}/user_features.csv")
    labels = pd.read_csv(f"{DATA_DIR}/labels.csv")

    feature_cols = [c for c in features.columns if c != "user_id"]
    X = standardize(features[feature_cols].values)

    merged = features.merge(labels[["user_id", "is_malicious", "scenario"]], on="user_id", how="left")
    y_true = merged["is_malicious"].fillna(False).astype(int).values
    k = int(y_true.sum())  # flag as many as we know are malicious

    # --- Isolation Forest (from scratch) ---
    iso = IsolationForestScratch(n_trees=300, sample_size=64, random_state=42)
    iso.fit(X)
    iso_scores = iso.decision_function(X)

    # --- Mahalanobis detector (One-Class SVM stand-in) ---
    maha = MahalanobisDetector()
    maha.fit(X)
    maha_scores = maha.decision_function(X)

    # --- Day 3 naive z-score baseline, recomputed here for a fair 3-way comparison ---
    z_scores = np.abs(X).sum(axis=1)

    results = {}
    for name, scores in [
        ("Day3_zscore_baseline", z_scores),
        ("IsolationForest_scratch", iso_scores),
        ("Mahalanobis_detector", maha_scores),
    ]:
        precision, recall, f1 = precision_recall_f1_at_k(y_true, scores, k)
        auc = roc_auc(y_true, scores)
        results[name] = {"precision@k": precision, "recall@k": recall, "f1@k": f1, "roc_auc": auc}

    results_df = pd.DataFrame(results).T
    results_df.index.name = "model"
    results_df.to_csv(f"{DATA_DIR}/day4_model_comparison.csv")

    scored = merged[["user_id", "is_malicious", "scenario"]].copy()
    scored["iso_forest_score"] = iso_scores
    scored["mahalanobis_score"] = maha_scores
    scored["day3_zscore"] = z_scores
    scored = scored.sort_values("iso_forest_score", ascending=False)
    scored.to_csv(f"{DATA_DIR}/day4_anomaly_scores.csv", index=False)

    # --- ROC curve plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    for name, scores in [
        ("Isolation Forest (scratch)", iso_scores),
        ("Mahalanobis detector", maha_scores),
        ("Day 3 z-score baseline", z_scores),
    ]:
        fpr, tpr = roc_curve_points(y_true, scores)
        auc_val = roc_auc(y_true, scores)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Day 4: ROC curves vs. ground truth")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig("docs/figures/day4_roc_curves.png", dpi=120)
    plt.close()

    print(f"k (flagged users) = {k}\n")
    print(results_df.round(3).to_string())
    print(f"\nTop {k} by Isolation Forest score:")
    print(scored.head(k)[["user_id", "iso_forest_score", "is_malicious", "scenario"]].to_string(index=False))

    return results_df, scored


if __name__ == "__main__":
    main()
