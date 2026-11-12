"""SHAP / LIME (real libraries + scratch) and the graph layer."""
import numpy as np
import pandas as pd

import dashboard_utils as du
from explain.real_explainers import lime_values_real, shap_values_real
from pyvis_graph import build_pyvis_html


def _first_insider(data, scenario):
    return data["labels"][data["labels"].scenario == scenario].user_id.iloc[0]


def test_real_shap_is_additive(models, matrix):
    X, cols, *_ = matrix
    model = models["Isolation Forest (sklearn)"]
    x, bg = X[0], X.mean(axis=0)
    phi = shap_values_real(model.decision_function, x, bg, n_samples=2000)
    assert phi.shape == (len(cols),)
    expected = model.decision_function(x[None])[0] - model.decision_function(bg[None])[0]
    assert abs(phi.sum() - expected) < 1e-3  # SHAP efficiency property


def test_shap_explains_after_hours_insider_with_logon_features(data, models, matrix):
    X, cols, *_ = matrix
    uid = _first_insider(data, "after_hours_login")
    shap_s, lime_s = du.explain_user(models, "Isolation Forest (sklearn)", X, cols, data["features"].user_id.values, uid, engine="library")
    top = set(shap_s.index[:6])
    assert top & {"after_hours_logon_count", "after_hours_logon_ratio", "avg_logon_hour", "std_logon_hour"}, top


def test_shap_explains_exfil_email_insider_with_email_features(data, models, matrix):
    X, cols, *_ = matrix
    uid = _first_insider(data, "data_exfil_email")
    shap_s, _ = du.explain_user(models, "Isolation Forest (sklearn)", X, cols, data["features"].user_id.values, uid, engine="library")
    top = set(shap_s.index[:6])
    assert top & {"max_external_size_kb", "avg_email_size_kb", "external_email_count", "total_attachments", "external_email_ratio"}, top


def test_explain_engines_both_work_and_agree_on_sign_of_top_feature(data, models, matrix):
    X, cols, *_ = matrix
    uid = _first_insider(data, "mass_file_access")
    ids = data["features"].user_id.values
    lib, _ = du.explain_user(models, "Isolation Forest (sklearn)", X, cols, ids, uid, engine="library")
    scr, _ = du.explain_user(models, "Isolation Forest (sklearn)", X, cols, ids, uid, engine="scratch")
    assert len(lib) == len(scr) == len(cols)
    assert np.sign(lib.iloc[0]) == np.sign(scr[lib.index[0]])


def test_lime_returns_one_weight_per_feature(models, matrix):
    X, cols, *_ = matrix
    w = lime_values_real(models["Isolation Forest (sklearn)"].decision_function, X[1], X, cols, num_samples=300)
    assert w.shape == (len(cols),)


def test_graph_has_all_users_and_resources(data):
    G = du.build_full_graph(data)
    users = [n for n, d in G.nodes(data=True) if d.get("node_type") == "user"]
    assert len(users) == len(data["users"])
    assert any(str(n).startswith("folder:") for n in G.nodes)


def test_pyvis_html_is_self_contained_and_marks_insiders(data, matrix):
    G = du.build_full_graph(data)
    html = build_pyvis_html(G, {u: 50.0 for u in data["users"].user_id}, data["labels"])
    assert "vis-network" in html or "vis.Network" in html
    assert "GROUND TRUTH" in html
    focused = build_pyvis_html(G, {}, data["labels"], focus_user="U0027")
    assert len(focused) < len(html)
