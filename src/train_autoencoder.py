"""
Day 5 (part 1) - Autoencoder anomaly detection.

Trains the from-scratch Autoencoder (src/models/autoencoder.py) on the
label-free user_features.csv, scores every user by reconstruction
error, and extends Day 4's model comparison table with this fourth
approach.
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from models.autoencoder import AutoencoderScratch
from train_baseline_models import (
    standardize, roc_auc, precision_recall_f1_at_k, roc_curve_points,
)

DATA_DIR = "data/processed/full"


def main():
    features = pd.read_csv(f"{DATA_DIR}/user_features.csv")
    labels = pd.read_csv(f"{DATA_DIR}/labels.csv")
    prior = pd.read_csv(f"{DATA_DIR}/day4_model_comparison.csv", index_col="model")

    feature_cols = [c for c in features.columns if c != "user_id"]
    X = standardize(features[feature_cols].values)

    merged = features.merge(labels[["user_id", "is_malicious", "scenario"]], on="user_id", how="left")
    y_true = merged["is_malicious"].fillna(False).astype(int).values
    k = int(y_true.sum())

    ae = AutoencoderScratch(hidden_dim=16, bottleneck_dim=2, lr=0.03, epochs=200, random_state=42)
    ae.fit(X)
    ae_scores = ae.decision_function(X)

    precision, recall, f1 = precision_recall_f1_at_k(y_true, ae_scores, k)
    auc = roc_auc(y_true, ae_scores)
    prior.loc["Autoencoder_scratch"] = [precision, recall, f1, auc]
    prior.to_csv(f"{DATA_DIR}/day4_model_comparison.csv")  # now covers Day 4 + 5 models

    scored = pd.read_csv(f"{DATA_DIR}/day4_anomaly_scores.csv")
    scored["autoencoder_score"] = ae_scores
    scored.to_csv(f"{DATA_DIR}/day4_anomaly_scores.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ae.loss_history_)
    axes[0].set_title("Autoencoder training loss (MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Reconstruction MSE")

    fpr, tpr = roc_curve_points(y_true, ae_scores)
    axes[1].plot(fpr, tpr, label=f"Autoencoder (AUC={auc:.3f})")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("Autoencoder ROC curve")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("docs/figures/day5_autoencoder.png", dpi=120)
    plt.close()

    print(f"Final training loss: {ae.loss_history_[-1]:.4f}")
    print(f"precision@{k}={precision:.3f} recall@{k}={recall:.3f} f1@{k}={f1:.3f} auc={auc:.3f}\n")
    print("Updated model comparison table:")
    print(prior.round(3).to_string())


if __name__ == "__main__":
    main()
