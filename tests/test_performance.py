"""Responsiveness guards - these are the things that made the dashboard feel 'stuck'."""
import time

import pytest

import dashboard_utils as du
from pyvis_graph import build_pyvis_html


def test_models_train_lazily(matrix):
    lazy = du.LazyModels(matrix[0])
    assert len(lazy) == 0
    t = time.time()
    lazy["Isolation Forest (sklearn)"]
    assert len(lazy) == 1 and time.time() - t < 10
    with pytest.raises(KeyError):
        lazy["not a model"]


@pytest.mark.parametrize("name", du.MODEL_CHOICES)
def test_explanation_is_interactive_speed(data, models, matrix, name):
    X, cols, *_ = matrix
    t = time.time()
    du.explain_user(models, name, X, cols, data["features"].user_id.values, "U0027")
    assert time.time() - t < 15, f"{name} explanation too slow"


def test_whole_org_graph_is_sparse_enough_to_read(data):
    G = du.build_full_graph(data)
    html = build_pyvis_html(G, {}, data["labels"])
    n_edges = html.count('"from"')
    assert n_edges < 900, n_edges  # 5,000+ edges was an unreadable hairball
    denser = build_pyvis_html(G, {}, data["labels"], min_edge_weight=2).count('"from"')
    assert denser > n_edges


def test_whole_org_graph_labels_only_risky_users(data):
    G = du.build_full_graph(data)
    risk = {u: 10.0 for u in data["users"].user_id}
    html = build_pyvis_html(G, risk, data["labels"])
    # low-risk users are unlabelled, insiders (black outline) keep their label
    assert '"label": "U0027"' in html
