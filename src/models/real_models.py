"""
Real-library detectors named in the project proposal.

All classes share the interface used everywhere else in this repo:
    .fit(X)               -> self
    .decision_function(X) -> 1-D array, HIGHER = MORE ANOMALOUS

(scikit-learn's own convention is the opposite - higher = more normal -
so the sign is flipped here to stay compatible with the dashboard,
the evaluation code and the from-scratch models.)
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


class SklearnIsolationForest:
    def __init__(self, n_estimators=300, contamination="auto", random_state=42):
        self.model = IsolationForest(
            n_estimators=n_estimators, contamination=contamination, random_state=random_state
        )

    def fit(self, X):
        self.model.fit(X)
        return self

    def decision_function(self, X):
        return -self.model.decision_function(X)


class SklearnOneClassSVM:
    """gamma=None -> 0.15 / n_features. scikit-learn's default gamma="scale"
    is too narrow for ~28 standardized features and 100-200 users (AUC 0.76
    on the dev dataset vs ~1.0 with a wider kernel; see docs/robust_evaluation.md)."""

    def __init__(self, nu=0.1, gamma=None, kernel="rbf"):
        self.nu, self.gamma, self.kernel = nu, gamma, kernel
        self.model = None

    def fit(self, X):
        gamma = self.gamma if self.gamma is not None else 0.15 / X.shape[1]
        self.model = OneClassSVM(nu=self.nu, gamma=gamma, kernel=self.kernel)
        self.model.fit(X)
        return self

    def decision_function(self, X):
        return -self.model.decision_function(X)


class KerasAutoencoder:
    """Dense autoencoder; anomaly score = per-user reconstruction error (MSE).
    epochs is deliberately small: with only ~100-200 users a longer-trained
    autoencoder memorises the outliers and AUC collapses (0.95 @10 epochs ->
    0.38 @100 epochs on the dev dataset; see docs/robust_evaluation.md).

    clip: winsorised training. Training inputs are clipped to +/- clip sigma so
    the few extreme insiders do not dominate the loss and get memorised; scoring
    always uses the raw (unclipped) data. On 5 fresh seeds this raised mean
    ROC-AUC from 0.89 to 0.97. Set clip=None to disable."""

    def __init__(self, hidden_dim=16, bottleneck_dim=4, epochs=15, batch_size=16, random_state=42, clip=3.0):
        self.clip = clip
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None

    def fit(self, X):
        import tensorflow as tf

        tf.keras.utils.set_random_seed(self.random_state)
        X = np.asarray(X, dtype="float32")
        if self.clip is not None:
            X = np.clip(X, -self.clip, self.clip)
        d = X.shape[1]
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(d,)),
                tf.keras.layers.Dense(self.hidden_dim, activation="relu"),
                tf.keras.layers.Dense(self.bottleneck_dim, activation="relu"),
                tf.keras.layers.Dense(self.hidden_dim, activation="relu"),
                tf.keras.layers.Dense(d, activation="linear"),
            ]
        )
        self.model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss="mse")
        self.model.fit(X, X, epochs=self.epochs, batch_size=self.batch_size, verbose=0, shuffle=True)
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype="float32")
        recon = self.model.predict(X, verbose=0)
        return ((X - recon) ** 2).mean(axis=1)


def tensorflow_available():
    # find_spec avoids importing TensorFlow (hundreds of MB of RAM) just to check availability
    import importlib.util
    try:
        return importlib.util.find_spec("tensorflow") is not None
    except Exception:
        return False
