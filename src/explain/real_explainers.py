"""
Explanations with the REAL `shap` and `lime` libraries (the from-scratch
versions in kernel_shap.py / lime_explainer.py remain as a fallback and
as a cross-check).
"""

import numpy as np


def shap_values_real(predict_fn, x, background, n_samples=500, seed=42):
    """KernelExplainer SHAP values for one standardized sample `x`.
    `background` is a 2-D reference array (here: the population mean row)."""
    import shap

    np.random.seed(seed)
    bg = np.atleast_2d(background)
    explainer = shap.KernelExplainer(predict_fn, bg)
    vals = explainer.shap_values(np.atleast_2d(x), nsamples=n_samples, silent=True)
    return np.asarray(vals).reshape(-1)


def lime_values_real(predict_fn, x, training_data, feature_names, num_samples=800, seed=42):
    """LIME (tabular, regression mode) weights for one sample, returned in
    `feature_names` order."""
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        training_data=training_data,
        feature_names=list(feature_names),
        mode="regression",
        discretize_continuous=False,
        random_state=seed,
    )
    exp = explainer.explain_instance(
        x, predict_fn, num_features=len(feature_names), num_samples=num_samples
    )
    weights = dict(exp.as_map()[1])
    return np.array([weights.get(i, 0.0) for i in range(len(feature_names))])
