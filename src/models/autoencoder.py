"""
Minimal from-scratch Autoencoder, pure NumPy - no TensorFlow dependency.

Why hand-rolled: as with Isolation Forest / One-Class SVM in Day 4,
this dev sandbox's network proxy could not install large ML packages
(TensorFlow is 200+ MB, far larger than the scipy wheel that already
failed to download at ~40-50 KB/s). A plain feedforward autoencoder is
a small enough architecture that manual forward/backward passes in
NumPy reproduce the real method faithfully: an encoder compresses each
user's feature vector to a small bottleneck, a decoder reconstructs it,
and the reconstruction error (how badly a user's own behaviour
reconstructs after compression) is the anomaly score - exactly the
same idea TensorFlow's Keras Autoencoder would give you.

On a normal internet connection, `pip install tensorflow` (or the much
lighter `torch` for CPU use) works fine - swap this out for a real
Keras/PyTorch autoencoder there if you want GPU-ready training; this
class exposes the same `.fit()` / `.decision_function()` interface as
the other Day 4/5 models for a drop-in replacement.
"""

import numpy as np


def _relu(x):
    return np.maximum(0, x)


def _relu_grad(x):
    return (x > 0).astype(float)


class AutoencoderScratch:
    def __init__(self, hidden_dim=12, bottleneck_dim=6, lr=0.01, epochs=800, random_state=0):
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state
        self.weights_ = None
        self.loss_history_ = []

    def _init_weights(self, n_features):
        rng = np.random.RandomState(self.random_state)

        def layer(fan_in, fan_out):
            limit = np.sqrt(6 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, size=(fan_in, fan_out))

        h, b = self.hidden_dim, self.bottleneck_dim
        self.weights_ = {
            "W1": layer(n_features, h), "b1": np.zeros(h),
            "W2": layer(h, b), "b2": np.zeros(b),
            "W3": layer(b, h), "b3": np.zeros(h),
            "W4": layer(h, n_features), "b4": np.zeros(n_features),
        }

    def _forward(self, X):
        w = self.weights_
        z1 = X @ w["W1"] + w["b1"]
        a1 = _relu(z1)
        z2 = a1 @ w["W2"] + w["b2"]
        a2 = _relu(z2)  # bottleneck
        z3 = a2 @ w["W3"] + w["b3"]
        a3 = _relu(z3)
        z4 = a3 @ w["W4"] + w["b4"]  # linear output (reconstruction)
        cache = (X, z1, a1, z2, a2, z3, a3, z4)
        return z4, cache

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n, n_features = X.shape
        self._init_weights(n_features)
        w = self.weights_

        for epoch in range(self.epochs):
            recon, cache = self._forward(X)
            _, z1, a1, z2, a2, z3, a3, z4 = cache

            error = recon - X  # dL/d(recon), MSE loss
            loss = np.mean(error ** 2)
            self.loss_history_.append(loss)

            dW4 = a3.T @ error / n
            db4 = error.mean(axis=0)

            da3 = error @ w["W4"].T
            dz3 = da3 * _relu_grad(z3)
            dW3 = a2.T @ dz3 / n
            db3 = dz3.mean(axis=0)

            da2 = dz3 @ w["W3"].T
            dz2 = da2 * _relu_grad(z2)
            dW2 = a1.T @ dz2 / n
            db2 = dz2.mean(axis=0)

            da1 = dz2 @ w["W2"].T
            dz1 = da1 * _relu_grad(z1)
            dW1 = X.T @ dz1 / n
            db1 = dz1.mean(axis=0)

            w["W4"] -= self.lr * dW4; w["b4"] -= self.lr * db4
            w["W3"] -= self.lr * dW3; w["b3"] -= self.lr * db3
            w["W2"] -= self.lr * dW2; w["b2"] -= self.lr * db2
            w["W1"] -= self.lr * dW1; w["b1"] -= self.lr * db1

        return self

    def decision_function(self, X):
        """Reconstruction error (mean squared) per sample; higher = more anomalous."""
        X = np.asarray(X, dtype=float)
        recon, _ = self._forward(X)
        return np.mean((recon - X) ** 2, axis=1)
