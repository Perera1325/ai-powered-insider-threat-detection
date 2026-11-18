"""
Day 7 - Core logic for the Streamlit SOC dashboard, kept separate from
app.py so it can be tested directly with plain Python (see
tests/smoke_test_dashboard.py) without needing Streamlit installed.

app.py imports these functions and wires them into st.* UI calls - the
functions themselves contain all the actual data loading, scoring, and
plotting logic.
"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from models.isolation_forest import IsolationForestScratch
from models.autoencoder import AutoencoderScratch
from models.mahalanobis import MahalanobisDetector
from models.real_models import KerasAutoencoder, SklearnIsolationForest, SklearnOneClassSVM, tensorflow_available
from explain.real_explainers import lime_values_real, shap_values_real
from train_baseline_models import standardize
from explain.kernel_shap import kernel_shap_values
from explain.lime_explainer import lime_explain
from graph_analysis import build_graph, _folder_category  # noqa: F401 (re-exported)

DATA_DIR = "data/processed/full"

SCRATCH_MODELS = ["Isolation Forest", "Autoencoder", "Mahalanobis detector"]
REAL_MODELS = ["Isolation Forest (sklearn)", "One-Class SVM (sklearn)", "Autoencoder (Keras)"]
# real-library models first (the ones named in the proposal); Keras only if TensorFlow is installed
MODEL_CHOICES = [m for m in REAL_MODELS + SCRATCH_MODELS if m != "Autoencoder (Keras)" or tensorflow_available()]


def load_data():
    """Loads everything the dashboard needs. Returns a dict of DataFrames."""
    features = pd.read_csv(f"{DATA_DIR}/user_features.csv")
    labels = pd.read_csv(f"{DATA_DIR}/labels.csv")
    users = pd.read_csv(f"{DATA_DIR}/users.csv")
    file_df = pd.read_csv(f"{DATA_DIR}/file.csv", parse_dates=["timestamp"])
    email_df = pd.read_csv(f"{DATA_DIR}/email.csv", parse_dates=["timestamp"])
    model_comparison = pd.read_csv(f"{DATA_DIR}/day4_model_comparison.csv", index_col="model")
    return {
        "features": features,
        "labels": labels,
        "users": users,
        "file_df": file_df,
        "email_df": email_df,
        "model_comparison": model_comparison,
    }


def get_feature_matrix(features):
    feature_cols = [c for c in features.columns if c != "user_id"]
    raw = features[feature_cols].values
    mean = raw.mean(axis=0)
    std = raw.std(axis=0)
    std_safe = np.where(std == 0, 1, std)
    X = (raw - mean) / std_safe
    return X, feature_cols, mean, std


def train_models(X):
    """Trains all three Day 4/5 models once; cache the result in the app
    layer (st.cache_resource) so this doesn't re-run on every interaction."""
    iso = IsolationForestScratch(n_trees=300, sample_size=64, random_state=42).fit(X)
    ae = AutoencoderScratch(hidden_dim=16, bottleneck_dim=2, lr=0.03, epochs=200, random_state=42).fit(X)
    maha = MahalanobisDetector().fit(X)
    models = {
        "Isolation Forest (sklearn)": SklearnIsolationForest().fit(X),
        "One-Class SVM (sklearn)": SklearnOneClassSVM().fit(X),
        "Isolation Forest": iso,
        "Autoencoder": ae,
        "Mahalanobis detector": maha,
    }
    if tensorflow_available():
        models["Autoencoder (Keras)"] = KerasAutoencoder().fit(X)
    return {k: models[k] for k in MODEL_CHOICES if k in models}


class LazyModels(dict):
    """Trains each detector the first time it is selected (instead of all six at
    startup: ~25 s -> ~1 s to first render). Behaves like the dict from train_models()."""

    def __init__(self, X):
        super().__init__()
        self._X = X

    def __missing__(self, name):
        if name not in MODEL_CHOICES:
            raise KeyError(name)
        self[name] = _train_one(name, self._X)
        return self[name]

    def __contains__(self, name):
        return name in MODEL_CHOICES


def _train_one(name, X):
    if name == "Isolation Forest (sklearn)":
        return SklearnIsolationForest().fit(X)
    if name == "One-Class SVM (sklearn)":
        return SklearnOneClassSVM().fit(X)
    if name == "Autoencoder (Keras)":
        return KerasAutoencoder().fit(X)
    if name == "Isolation Forest":
        return IsolationForestScratch(n_trees=300, sample_size=64, random_state=42).fit(X)
    if name == "Autoencoder":
        return AutoencoderScratch(hidden_dim=16, bottleneck_dim=2, lr=0.03, epochs=200, random_state=42).fit(X)
    if name == "Mahalanobis detector":
        return MahalanobisDetector().fit(X)
    raise KeyError(name)


def score_users(models, model_name, X):
    model = models[model_name]
    scores = model.decision_function(X)
    # normalize every model's score to a 0-100 "risk score" for consistent display
    lo, hi = scores.min(), scores.max()
    risk_score = 100 * (scores - lo) / (hi - lo + 1e-9)
    return scores, risk_score


def risk_level(risk_score):
    if risk_score >= 70:
        return "High"
    if risk_score >= 40:
        return "Medium"
    return "Low"


def build_alert_table(features, labels, models, model_name, X, top_k=15):
    scores, risk_score = score_users(models, model_name, X)
    df = features[["user_id"]].copy()
    df["anomaly_score"] = scores
    df["risk_score"] = risk_score.round(1)
    df["risk_level"] = df.risk_score.apply(risk_level)
    df = df.merge(labels[["user_id", "is_malicious", "scenario"]], on="user_id", how="left")
    df["is_malicious"] = df["is_malicious"].fillna(False)
    df = df.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df.head(top_k)


def explain_user(models, model_name, X, feature_cols, user_ids, uid, n_shap_samples=800, n_lime_samples=600,
                 engine="library"):
    """engine='library' uses the real `shap` / `lime` packages (falls back to the
    from-scratch implementations if they are not installed); 'scratch' forces the
    from-scratch versions."""
    model = models[model_name]
    predict_fn = model.decision_function
    x_idx = np.where(user_ids == uid)[0][0]
    x = X[x_idx]
    background = X.mean(axis=0)
    feature_std = np.ones(len(feature_cols))

    if engine == "library":
        try:
            phi = shap_values_real(predict_fn, x, background, n_samples=n_shap_samples)
            lime_coef = lime_values_real(predict_fn, x, X, feature_cols, num_samples=n_lime_samples)
        except ImportError:
            engine = "scratch"
    if engine != "library":
        phi, base_value = kernel_shap_values(predict_fn, x, background, n_samples=n_shap_samples, random_state=42)
        lime_coef, _ = lime_explain(predict_fn, x, feature_std, n_samples=n_lime_samples, random_state=42)

    shap_series = pd.Series(phi, index=feature_cols).sort_values(key=abs, ascending=False)
    lime_series = pd.Series(lime_coef, index=feature_cols).sort_values(key=abs, ascending=False)
    return shap_series, lime_series


def explanation_figure(shap_series, lime_series, uid, top_n=8):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    shap_top = shap_series.head(top_n).sort_values()
    lime_top = lime_series.head(top_n).sort_values()

    shap_top.plot.barh(ax=axes[0], color=np.where(shap_top >= 0, "#d94a4a", "#4a90d9"))
    axes[0].set_title(f"{uid} - SHAP")
    lime_top.plot.barh(ax=axes[1], color=np.where(lime_top >= 0, "#d94a4a", "#4a90d9"))
    axes[1].set_title(f"{uid} - LIME")
    plt.tight_layout()
    return fig


def build_full_graph(data):
    return build_graph(data["file_df"], data["email_df"], data["users"])


def subgraph_figure(G, uid, radius=1, top_contacts=12):
    """Clean 'ego' view: the user in the centre, shared folders on an inner ring and the
    strongest email contacts on an outer ring (readable, unlike the full neighbourhood)."""
    if uid not in G:
        return None
    import math

    resources = [n for n in G.neighbors(uid) if G.nodes[n].get("node_type") == "resource"]
    people = [(n, G[uid][n].get("weight", 1)) for n in G.neighbors(uid) if n not in resources]
    people = [n for n, _ in sorted(people, key=lambda t: -t[1])[:top_contacts]]

    fig, ax = plt.subplots(figsize=(7, 6))
    pos = {uid: (0.0, 0.0)}
    for i, n in enumerate(resources):
        a = 2 * math.pi * i / max(len(resources), 1) + math.pi / 6
        pos[n] = (0.45 * math.cos(a), 0.45 * math.sin(a))
    for i, n in enumerate(people):
        a = 2 * math.pi * i / max(len(people), 1)
        pos[n] = (math.cos(a), math.sin(a))
    for n in resources + people:
        w = G[uid][n].get("weight", 1)
        ax.plot([0, pos[n][0]], [0, pos[n][1]], color="#B6C2D4", lw=0.8 + min(float(w), 20) / 6, zorder=1, alpha=0.9)
    ax.scatter([pos[n][0] for n in resources], [pos[n][1] for n in resources], s=900, marker="s",
               color="#F59E0B", edgecolor="white", zorder=3)
    for n in resources:
        ax.text(*pos[n], str(n).replace("folder:", ""), ha="center", va="center", fontsize=8, color="#1B2433", zorder=4)
    ax.scatter([pos[n][0] for n in people], [pos[n][1] for n in people], s=420, color="#2563EB",
               edgecolor="white", zorder=3)
    for n in people:
        x, y = pos[n]
        ax.text(x * 1.16, y * 1.16, n, ha="center", va="center", fontsize=8.5, color="#1B2433")
    ax.scatter([0], [0], s=1100, color="#DC2626", edgecolor="white", linewidth=2, zorder=5)
    ax.text(0, -0.17, uid, ha="center", va="center", fontsize=10, fontweight="bold", color="#1B2433", zorder=6)
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.3, 1.3)
    ax.set_title(f"{uid}: shared folders (inner ring) and {len(people)} strongest email contacts")
    ax.axis("off")
    plt.tight_layout()
    return fig