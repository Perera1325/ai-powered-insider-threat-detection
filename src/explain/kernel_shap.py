"""
Simplified from-scratch Kernel SHAP, pure NumPy - no `shap` package.

Why hand-rolled: the real `shap` library depends on scikit-learn/scipy,
which this dev sandbox could not install for the same bandwidth reasons
documented in Day 4/5 (docs/day4_progress.md, docs/day5_progress.md).
`shap` itself downloads fine (~1MB); it's the scipy/sklearn dependency
chain underneath that stalls.

This implements the core idea of Kernel SHAP (Lundberg & Lee, 2017):
  1. Sample random feature "coalitions" (binary masks) of varying size.
  2. For each coalition, build a hybrid instance that keeps the
     explained instance's values for "in" features and swaps in
     background values for "out" features, and score it with the model.
  3. Fit a weighted linear regression (SHAP kernel weighting, which
     up-weights very small and very large coalitions) to recover each
     feature's additive contribution.
  4. Rescale contributions so they sum exactly to f(x) - f(background)
     (the Shapley "efficiency" property) - a simplification of the
     constrained solve used in the reference implementation, applied
     as a final correction step rather than a hard constraint in the
     regression itself.

This is a faithful-in-spirit, simplified implementation for an
educational/from-scratch context - not a bit-for-bit reproduction of
the reference `shap` library. Swap in `shap.KernelExplainer` when
running with normal internet access for a more precise, established
implementation on higher-dimensional feature sets.
"""

import numpy as np


def _shap_kernel_weight(d, z_size):
    if z_size == 0 or z_size == d:
        return 10_000.0  # large weight to anchor the endpoints
    from math import comb
    return (d - 1) / (comb(d, z_size) * z_size * (d - z_size))


def kernel_shap_values(predict_fn, x, background, n_samples=1500, random_state=0):
    """
    predict_fn: callable, takes (N, d) array -> (N,) scores
    x: (d,) instance to explain
    background: (d,) reference/background vector (e.g. population mean)
    Returns: (phi, base_value) where phi is a (d,) array of per-feature
    contributions summing to predict_fn(x) - base_value.
    """
    rng = np.random.RandomState(random_state)
    x = np.asarray(x, dtype=float)
    background = np.asarray(background, dtype=float)
    d = x.shape[0]

    base_value = float(predict_fn(background[None, :])[0])
    f_x = float(predict_fn(x[None, :])[0])
    total_effect = f_x - base_value

    Z, weights, y = [], [], []
    for _ in range(n_samples):
        z_size = rng.randint(1, d)  # exclude 0 and d, handled as anchors below
        idx = rng.choice(d, size=z_size, replace=False)
        z = np.zeros(d)
        z[idx] = 1
        hybrid = z * x + (1 - z) * background
        score = float(predict_fn(hybrid[None, :])[0])
        Z.append(z)
        weights.append(_shap_kernel_weight(d, z_size))
        y.append(score - base_value)

    Z = np.array(Z)
    weights = np.array(weights)
    y = np.array(y)

    # weighted least squares: minimize sum(w * (y - Z @ phi)^2), no intercept
    # (intercept is implicitly base_value, already subtracted out of y)
    W = np.diag(weights)
    ZtW = Z.T @ W
    phi = np.linalg.lstsq(ZtW @ Z, ZtW @ y, rcond=None)[0]

    # efficiency correction: force sum(phi) == total_effect exactly
    correction = (total_effect - phi.sum()) / d
    phi = phi + correction

    return phi, base_value
