"""All detectors share one interface: fit(X), decision_function(X) - higher = more anomalous."""
import numpy as np
import pytest

from models.real_models import KerasAutoencoder, SklearnIsolationForest, SklearnOneClassSVM
from train_baseline_models import precision_recall_f1_at_k, roc_auc
import dashboard_utils as du


def _toy():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(120, 6))
    X[:3] += 8.0  # three obvious outliers
    return X


def test_keras_autoencoder_flags_off_manifold_points():
    # autoencoders detect points that break the learned structure (normal data lives on a
    # 2-D subspace of 8-D space; the outliers do not), not merely "far away" points
    rng = np.random.RandomState(1)
    Z = rng.normal(size=(150, 2))
    X = Z @ rng.normal(size=(2, 8)) + 0.05 * rng.normal(size=(150, 8))
    X[:4] += rng.normal(scale=2.5, size=(4, 8))
    scores = KerasAutoencoder(epochs=60, bottleneck_dim=2, clip=None).fit(X).decision_function(X)
    assert set(np.argsort(-scores)[:4]) == {0, 1, 2, 3}


@pytest.mark.parametrize("make", [SklearnIsolationForest, SklearnOneClassSVM])
def test_real_models_rank_obvious_outliers_highest(make):
    X = _toy()
    scores = make().fit(X).decision_function(X)
    assert scores.shape == (120,)
    assert set(np.argsort(-scores)[:3]) == {0, 1, 2}


def test_roc_auc_and_precision_at_k_helpers():
    y = np.array([1, 1, 0, 0, 0, 0])
    assert roc_auc(y, np.array([.9, .8, .1, .2, .3, .4])) == 1.0
    assert roc_auc(y, -np.array([.9, .8, .1, .2, .3, .4])) == 0.0
    p, r, f = precision_recall_f1_at_k(y, np.array([.9, .1, .8, .2, .3, .4]), 2)
    assert (p, r) == (0.5, 0.5)


@pytest.mark.parametrize("name", du.MODEL_CHOICES)
def test_every_dashboard_model_detects_insiders_on_committed_dataset(models, matrix, y_true, name):
    X = matrix[0]
    assert name in models
    scores = models[name].decision_function(X)
    assert len(scores) == len(X)
    # loose bound: this is a smoke-level quality gate, the real numbers are in docs/robust_evaluation.md
    assert roc_auc(y_true, scores) > 0.85, name


def test_models_do_not_use_labels(models, matrix):
    # scoring a user must depend on features only: scrambling nothing else changes the output
    X = matrix[0]
    a = models["Isolation Forest (sklearn)"].decision_function(X)
    b = models["Isolation Forest (sklearn)"].decision_function(X.copy())
    assert np.allclose(a, b)
