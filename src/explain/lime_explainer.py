"""
Simplified from-scratch LIME, pure NumPy - no `lime` package.

Same rationale as kernel_shap.py: the real `lime` package depends on
scikit-learn/scipy, which this sandbox could not install. LIME's core
idea (Ribeiro, Singh & Guestrin, 2016) is reproduced directly:

  1. Generate random perturbations of the instance being explained
     (Gaussian noise in standardized feature space).
  2. Score each perturbation with the (black-box) model.
  3. Weight each perturbed sample by its proximity to the original
     instance (an RBF/exponential kernel on distance).
  4. Fit a weighted, L2-regularized ("ridge") linear regression - the
     interpretable local surrogate. Its coefficients approximate how
     much each feature pushed the prediction up/down *near this
     specific instance*, which is what makes LIME "local" as opposed
     to SHAP's global-consistency guarantees.

Swap in the real `lime` package when running with normal internet
access if you want its exact discretization/sampling strategy;
this implementation follows the same mathematical recipe.
"""

import numpy as np


def lime_explain(predict_fn, x, feature_std, n_samples=1000, kernel_width=None,
                  ridge_alpha=1.0, random_state=0):
    """
    predict_fn: callable, takes (N, d) array -> (N,) scores
    x: (d,) instance to explain (already in standardized feature space)
    feature_std: (d,) per-feature std to scale the perturbation noise
                  (use 1.0 array if x is already standardized)
    Returns: (coef, intercept) - coef is a (d,) array of local linear
    weights; larger |coef_i| means feature i matters more for this
    instance's prediction, in this local neighbourhood.
    """
    rng = np.random.RandomState(random_state)
    x = np.asarray(x, dtype=float)
    d = x.shape[0]
    kernel_width = kernel_width or np.sqrt(d) * 0.75

    noise = rng.normal(loc=0, scale=0.5, size=(n_samples, d)) * feature_std
    samples = x[None, :] + noise
    samples = np.vstack([samples, x[None, :]])  # always include the instance itself

    scores = predict_fn(samples)

    dist = np.linalg.norm(samples - x[None, :], axis=1)
    weights = np.exp(-(dist ** 2) / (kernel_width ** 2))

    # weighted ridge regression with intercept: augment X with a bias column
    X_design = np.hstack([samples, np.ones((samples.shape[0], 1))])
    W = np.diag(weights)
    XtW = X_design.T @ W
    reg = ridge_alpha * np.eye(X_design.shape[1])
    reg[-1, -1] = 0  # don't regularize the intercept
    beta = np.linalg.lstsq(XtW @ X_design + reg, XtW @ scores, rcond=None)[0]

    coef, intercept = beta[:-1], beta[-1]
    return coef, intercept
