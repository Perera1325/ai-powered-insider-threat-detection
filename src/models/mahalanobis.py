"""
Mahalanobis-distance multivariate anomaly detector, pure NumPy.

Used in this project as a stand-in second unsupervised baseline in
place of scikit-learn's One-Class SVM, because the sandbox this
project was initially built in could not install scikit-learn's scipy
dependency (throttled outbound bandwidth on a large binary wheel - see
models/isolation_forest.py docstring for the measured details).

Conceptually it plays a similar role to One-Class SVM: it learns a
single "normal" region from the (unlabeled) training data and scores
new points by their distance from it - here, a covariance-aware
distance from the population mean, rather than an SVM's learned
maximum-margin boundary in kernel space.

Recommended: once running with normal internet access, install
scikit-learn and replace this with `sklearn.svm.OneClassSVM(kernel="rbf")`
for the non-linear, margin-based boundary the project proposal
specifies - this class is a documented, honest substitute, not a
claim of equivalence.
"""

import numpy as np


class MahalanobisDetector:
    def __init__(self, regularization=1e-6):
        self.regularization = regularization
        self.mean_ = None
        self.inv_cov_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        cov += np.eye(cov.shape[0]) * self.regularization  # numerical stability
        self.inv_cov_ = np.linalg.pinv(cov)
        return self

    def decision_function(self, X):
        """Returns the Mahalanobis distance; higher = more anomalous."""
        X = np.asarray(X, dtype=float)
        diff = X - self.mean_
        # sqrt(diff @ inv_cov @ diff.T), diagonal only
        dist_sq = np.einsum("ij,jk,ik->i", diff, self.inv_cov_, diff)
        return np.sqrt(np.clip(dist_sq, 0, None))
