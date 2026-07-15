"""
Minimal from-scratch Isolation Forest (Liu, Ting & Zhou, 2008), pure
NumPy - no scikit-learn/scipy dependency.

Why hand-rolled: this project's dev sandbox could not install
scikit-learn's scipy dependency (large binary wheel, throttled to a
few tens of KB/s on the sandbox's outbound proxy - verified by direct
curl timing). The algorithm itself is simple enough to implement
directly and matches the published method: build an ensemble of random
partitioning trees; points that are isolated in fewer splits (shorter
average path length across the ensemble) are more anomalous.

On a normal internet connection, `pip install scikit-learn` takes a
few seconds and `sklearn.ensemble.IsolationForest` can be dropped in
as a direct replacement (same idea, same output ranking) - recommended
once this project is run outside this constrained sandbox.
"""

import numpy as np


class _Node:
    __slots__ = ("feature", "split_value", "left", "right", "size", "is_leaf")

    def __init__(self, size, is_leaf=True):
        self.feature = None
        self.split_value = None
        self.left = None
        self.right = None
        self.size = size
        self.is_leaf = is_leaf


def _c(n):
    """Average path length of an unsuccessful search in a BST of n points."""
    if n <= 1:
        return 0.0
    return 2.0 * (np.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)


class IsolationForestScratch:
    def __init__(self, n_trees=200, sample_size=64, max_depth=None, random_state=0):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees_ = []
        self.c_sample_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.RandomState(self.random_state)
        n = X.shape[0]
        sub_n = min(self.sample_size, n)
        depth_limit = self.max_depth or int(np.ceil(np.log2(max(sub_n, 2))))
        self.c_sample_ = _c(sub_n)

        self.trees_ = []
        for i in range(self.n_trees):
            idx = rng.choice(n, size=sub_n, replace=False)
            sample = X[idx]
            tree = self._build_tree(sample, depth_limit, rng)
            self.trees_.append(tree)
        return self

    def _build_tree(self, X, depth_limit, rng, depth=0):
        n = X.shape[0]
        if depth >= depth_limit or n <= 1:
            return _Node(size=n, is_leaf=True)

        n_features = X.shape[1]
        # skip constant features
        candidate_feats = [f for f in range(n_features) if X[:, f].max() > X[:, f].min()]
        if not candidate_feats:
            return _Node(size=n, is_leaf=True)

        feature = rng.choice(candidate_feats)
        fmin, fmax = X[:, feature].min(), X[:, feature].max()
        split_value = rng.uniform(fmin, fmax)

        left_mask = X[:, feature] < split_value
        if left_mask.all() or (~left_mask).all():
            return _Node(size=n, is_leaf=True)

        node = _Node(size=n, is_leaf=False)
        node.feature = feature
        node.split_value = split_value
        node.left = self._build_tree(X[left_mask], depth_limit, rng, depth + 1)
        node.right = self._build_tree(X[~left_mask], depth_limit, rng, depth + 1)
        return node

    def _path_length(self, x, node, depth=0):
        if node.is_leaf:
            return depth + _c(node.size)
        if x[node.feature] < node.split_value:
            return self._path_length(x, node.left, depth + 1)
        return self._path_length(x, node.right, depth + 1)

    def decision_function(self, X):
        """Returns anomaly SCORE in [0,1]; higher = more anomalous."""
        X = np.asarray(X, dtype=float)
        scores = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            avg_path = np.mean([self._path_length(x, tree) for tree in self.trees_])
            scores[i] = 2 ** (-avg_path / self.c_sample_)
        return scores
