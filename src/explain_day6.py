"""
Day 6 - Explainable AI (SHAP + LIME) for flagged insider-threat alerts.

For each of the 6 known malicious users (plus one normal user for
contrast), explains the Isolation Forest anomaly score using both the
from-scratch Kernel SHAP and LIME implementations (src/explain/), and
checks whether the top contributing features make sense given the
scenario that was actually injected for that user back in Day 1 - i.e.
does the explanation match the ground truth story, not just the score.
"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from models.isolation_forest import IsolationForestScratch
from train_baseline_models import standardize
from explain.kernel_shap import kernel_shap_values
from explain.lime_explainer import lime_explain

DATA_DIR = "data/processed/full"
FIG_DIR = "docs/figures"


def main():
    features = pd.read_csv(f"{DATA_DIR}/user_features.csv")
    labels = pd.read_csv(f"{DATA_DIR}/labels.csv")
    feature_cols = [c for c in features.columns if c != "user_id"]

    X = standardize(features[feature_cols].values)
    user_ids = features.user_id.values

    iso = IsolationForestScratch(n_trees=300, sample_size=64, random_state=42)
    iso.fit(X)
    predict_fn = iso.decision_function

    background = X.mean(axis=0)
    feature_std = np.ones(len(feature_cols))  # X is already standardized

    malicious = labels[labels.is_malicious][["user_id", "scenario"]]
    # one normal user with a mid-range anomaly score, for contrast
    scores_all = predict_fn(X)
    normal_mask = ~np.isin(user_ids, malicious.user_id.values)
    normal_idx = np.argsort(np.abs(scores_all[normal_mask] - np.median(scores_all[normal_mask])))[0]
    normal_user_id = user_ids[normal_mask][normal_idx]

    targets = list(malicious.itertuples(index=False)) + [("__normal__", normal_user_id, "normal_baseline")]
    # fix tuple shape: (user_id, scenario)
    targets = [(row.user_id, row.scenario) for row in malicious.itertuples(index=False)]
    targets.append((normal_user_id, "normal_baseline"))

    report_lines = ["# Day 6 Explainability Report\n"]
    fig, axes = plt.subplots(len(targets), 2, figsize=(12, 3.2 * len(targets)))

    for row_i, (uid, scenario) in enumerate(targets):
        x_idx = np.where(user_ids == uid)[0][0]
        x = X[x_idx]

        phi, base_value = kernel_shap_values(predict_fn, x, background, n_samples=1500, random_state=42)
        lime_coef, lime_intercept = lime_explain(predict_fn, x, feature_std, n_samples=1000, random_state=42)

        shap_top = pd.Series(phi, index=feature_cols).sort_values(key=abs, ascending=False).head(6)
        lime_top = pd.Series(lime_coef, index=feature_cols).sort_values(key=abs, ascending=False).head(6)

        score = predict_fn(x[None, :])[0]
        report_lines.append(f"## {uid} — scenario: `{scenario}` (anomaly score={score:.3f})\n")
        report_lines.append("**Top SHAP contributors:**")
        report_lines.append("```")
        report_lines.append(shap_top.round(3).to_string())
        report_lines.append("```")
        report_lines.append("**Top LIME contributors:**")
        report_lines.append("```")
        report_lines.append(lime_top.round(3).to_string())
        report_lines.append("```\n")

        ax_shap, ax_lime = axes[row_i]
        shap_top.sort_values().plot.barh(ax=ax_shap, color=np.where(shap_top.sort_values() >= 0, "#d94a4a", "#4a90d9"))
        ax_shap.set_title(f"{uid} ({scenario}) — SHAP")
        lime_top.sort_values().plot.barh(ax=ax_lime, color=np.where(lime_top.sort_values() >= 0, "#d94a4a", "#4a90d9"))
        ax_lime.set_title(f"{uid} ({scenario}) — LIME")

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/day6_shap_lime_explanations.png", dpi=110)
    plt.close()

    with open("docs/day6_explainability_report.md", "w") as f:
        f.write("\n".join(report_lines) + "\n")

    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
