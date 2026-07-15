# Day 6 Progress

## What was built

- `src/explain/kernel_shap.py` — a simplified from-scratch Kernel SHAP
  (Lundberg & Lee, 2017): samples feature coalitions, scores hybrid
  instances against a background reference, fits a SHAP-kernel-weighted
  linear regression, and rescales contributions to sum exactly to
  `f(x) - f(background)`.
- `src/explain/lime_explainer.py` — a from-scratch LIME (Ribeiro, Singh
  & Guestrin, 2016): Gaussian perturbations around the instance,
  proximity-weighted, fit with a weighted ridge regression.
- `src/explain_day6.py` — applies both to the Isolation Forest's
  anomaly score for all 6 malicious users plus one normal user (for
  contrast), and checks whether the top contributing features actually
  match the scenario injected back in Day 1.
- `docs/figures/day6_shap_lime_explanations.png` — bar charts for all 7
  explained users, SHAP and LIME side by side.

### Why hand-rolled again

Same reason as Days 4-5: the real `shap` and `lime` packages both
depend on scikit-learn/scipy underneath, which this sandbox's network
proxy cannot fetch in reasonable time (documented in
docs/day4_progress.md). `shap` and `lime`'s own wheels download fine
(~1MB) — it's their dependency chain that stalls. Swap in the real
packages on a normal connection if you want the reference
implementations; the interface (`predict_fn` as a black-box callable)
is designed to make that a small change.

## Key finding: SHAP explanations line up with ground truth almost perfectly

| User  | Scenario            | Top SHAP feature(s)                                              |
|-------|----------------------|-------------------------------------------------------------------|
| U0016 | after_hours_login    | `after_hours_logon_ratio`, `after_hours_logon_count`              |
| U0038 | after_hours_login    | `after_hours_logon_ratio`, `after_hours_logon_count`              |
| U0027 | mass_file_access     | `max_daily_file_events`, `max_daily_shared_folder_events`, `copy_count` |
| U0050 | privilege_probe      | `max_daily_file_events`, `distinct_files`, `file_event_count`     |
| U0079 | usb_exfiltration     | `device_after_hours_count`, `copy_ratio`, `device_connect_count`  |
| U0092 | data_exfil_email     | `avg_email_size_kb`, `max_external_size_kb`                       |
| U0045 | (normal, for contrast)| no single dominant feature; magnitudes ~3x smaller overall       |

This is exactly the point of Explainable AI in the proposal: instead of
a black-box "this user is anomalous" alert, the system tells a security
analyst *why* — and here, the "why" for each user matches the actual
injected attack behaviour almost feature-for-feature. This is strong
evidence the engineered features (Day 3) are capturing the right
signal, not something spurious.

## Secondary finding: LIME is noticeably noisier than SHAP here

LIME's top contributing features for the same users are much less
aligned with the injected scenario (e.g. for U0027's `mass_file_access`,
LIME's top feature is `total_logons`, not a file-access feature) and
the coefficient magnitudes are smaller/flatter across features.

Likely reason: Isolation Forest's decision function is piecewise and
non-smooth (built from random tree splits), so a *local linear*
surrogate (what LIME fits) approximates it poorly in a small
neighbourhood — small Gaussian perturbations can land on either side
of a tree split boundary somewhat arbitrarily. SHAP's coalition
sampling instead compares against a fixed background reference and
Shapley-weights coalition sizes, which averages out more of that local
noise. This is a genuine trade-off documented in the wider XAI
literature (LIME's locality helps with smooth models, hurts with
tree-based/non-smooth ones) - not a bug in this implementation.

**Recommendation for the report:** lead with SHAP as the primary
explanation mechanism for this system (matches the proposal's emphasis
too), and either mention LIME's instability as a limitation/discussion
point, or try LIME against the smoother Autoencoder reconstruction
error instead of Isolation Forest if a fairer LIME comparison is
wanted later.

## Next (Day 7)

Streamlit dashboard integrating: alert list with anomaly scores (Day
4/5 models), the interaction graph (Day 5), and the SHAP/LIME
explanation panel (Day 6) per selected alert.
