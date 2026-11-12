"""
Robust evaluation (replaces the "everything scores 1.0" result of Day 4/5).

Experiment 1 - multi-seed generalisation
    Fresh datasets (unseen seeds, 200 users, 10 red-team insiders = 5%).
    Every model is trained on unlabeled features only; labels are used
    ONLY to score the ranking. Reports mean +/- std over seeds.

Experiment 2 - signal-strength stress test
    The injected users' feature vectors are shrunk toward the population
    mean: x' = mean + alpha * (x - mean). alpha=1 is the original attack,
    alpha=0.25 means an attack with a quarter of the deviation, i.e. a
    "low and slow" insider. Shows where each detector breaks down.

Run:  python3 src/robust_evaluation.py
"""

import os
import sys
import tempfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, "src")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import feature_engineering as fe
import generate_logs as gl
import inject_anomalies as ia
from models.autoencoder import AutoencoderScratch
from models.isolation_forest import IsolationForestScratch
from models.mahalanobis import MahalanobisDetector
from models.real_models import KerasAutoencoder, SklearnIsolationForest, SklearnOneClassSVM, tensorflow_available
from train_baseline_models import precision_recall_f1_at_k, roc_auc, standardize

SEEDS = [101, 102, 103, 104, 105, 106, 107, 108]
N_USERS = 200
N_MALICIOUS = 10
ALPHAS = [1.0, 0.75, 0.5, 0.35, 0.25]
OUT_CSV = "data/processed/full/robust_evaluation.csv"
OUT_DILUTION_CSV = "data/processed/full/robust_dilution.csv"


def model_zoo(seed):
    zoo = {
        "Z-score baseline": None,
        "Isolation Forest (scratch)": lambda: IsolationForestScratch(n_trees=300, sample_size=64, random_state=seed),
        "Isolation Forest (sklearn)": lambda: SklearnIsolationForest(random_state=seed),
        "One-Class SVM (sklearn)": lambda: SklearnOneClassSVM(),
        "Mahalanobis (scratch)": lambda: MahalanobisDetector(),
        "Autoencoder (scratch)": lambda: AutoencoderScratch(hidden_dim=16, bottleneck_dim=2, lr=0.03, epochs=200, random_state=seed),
    }
    if tensorflow_available():
        zoo["Autoencoder (Keras)"] = lambda: KerasAutoencoder(random_state=seed)
    return zoo


def score_all(X, seed):
    out = {}
    for name, factory in model_zoo(seed).items():
        out[name] = np.abs(X).sum(axis=1) if factory is None else factory().fit(X).decision_function(X)
    return out


def build_dataset(seed):
    with tempfile.TemporaryDirectory() as d:
        gl.main(out_dir=d, seed=seed, num_users=N_USERS)
        ia.main(data_dir=d, seed=seed, n_malicious=N_MALICIOUS)
        features, preview = fe.main(data_dir=d, write=False)
    cols = [c for c in features.columns if c != "user_id"]
    y = preview["is_malicious"].fillna(False).astype(int).values
    return features[cols].values.astype(float), y, cols


def _quiet(fn, *a, **k):
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def main():
    rows, dilution_rows = [], []
    for seed in SEEDS:
        raw, y, cols = _quiet(build_dataset, seed)
        X = standardize(raw)
        k = int(y.sum())
        for name, s in score_all(X, seed).items():
            p, r, f = precision_recall_f1_at_k(y, s, k)
            rows.append({"seed": seed, "model": name, "roc_auc": roc_auc(y, s), "precision@k": p, "recall@k": r, "f1@k": f})

        # Experiment 2: dilute the attack signal
        mean = raw.mean(axis=0)
        for alpha in ALPHAS:
            diluted = raw.copy()
            diluted[y == 1] = mean + alpha * (raw[y == 1] - mean)
            Xd = standardize(diluted)
            for name, s in score_all(Xd, seed).items():
                p, _, _ = precision_recall_f1_at_k(y, s, k)
                dilution_rows.append({"seed": seed, "alpha": alpha, "model": name, "roc_auc": roc_auc(y, s), "precision@k": p})
        print(f"seed {seed} done ({X.shape[0]} users, {k} insiders, {X.shape[1]} features)", flush=True)

    res = pd.DataFrame(rows)
    dil = pd.DataFrame(dilution_rows)
    res.to_csv(OUT_CSV, index=False)
    dil.to_csv(OUT_DILUTION_CSV, index=False)

    summary = res.groupby("model")[["roc_auc", "precision@k", "recall@k"]].agg(["mean", "std"]).round(3)
    print("\n=== Experiment 1: unseen seeds (mean/std over %d seeds) ===" % len(SEEDS))
    print(summary.to_string())

    dil_summary = dil.groupby(["model", "alpha"])["roc_auc"].mean().unstack().round(3)
    dil_summary = dil_summary[sorted(dil_summary.columns, reverse=True)]
    print("\n=== Experiment 2: mean ROC-AUC vs attack strength alpha ===")
    print(dil_summary.to_string())

    # Figures
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    order = res.groupby("model")["roc_auc"].mean().sort_values().index
    axes[0].barh(order, [res[res.model == m].roc_auc.mean() for m in order],
                 xerr=[res[res.model == m].roc_auc.std() for m in order], color="#4a90d9", capsize=3)
    axes[0].set_xlim(0, 1.05)
    axes[0].set_xlabel("ROC-AUC (mean +/- std, %d unseen seeds)" % len(SEEDS))
    axes[0].set_title("Generalisation to unseen datasets")
    for m in dil_summary.index:
        axes[1].plot(dil_summary.columns, dil_summary.loc[m], marker="o", label=m)
    axes[1].invert_xaxis()
    axes[1].set_xlabel("attack strength alpha (1 = full attack, 0.25 = low-and-slow)")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].set_title("Detection vs attack strength")
    axes[1].axhline(0.5, color="k", ls="--", alpha=0.3)
    axes[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("docs/figures/robust_evaluation.png", dpi=120)
    plt.close()

    md = ["# Robust evaluation (auto-generated by `src/robust_evaluation.py`)", "",
          f"Setup: {len(SEEDS)} unseen seeds, {N_USERS} users, {N_MALICIOUS} red-team insiders per dataset, ",
          "28 behavioural features. Labels are used only to score rankings, never to train.", "",
          "## Experiment 1 - ROC-AUC / precision@k / recall@k (mean, std over seeds)", "",
          summary.to_markdown(), "",
          "## Experiment 2 - mean ROC-AUC as the attack is weakened (alpha)", "",
          dil_summary.to_markdown(), "",
          "![robust](figures/robust_evaluation.png)", ""]
    with open("docs/robust_evaluation.md", "w") as f:
        f.write("\n".join(md))
    print("\nWrote docs/robust_evaluation.md and docs/figures/robust_evaluation.png")


if __name__ == "__main__":
    main()
