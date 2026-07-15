# 7-Day Build Plan

**Day 1 — Environment + dataset foundation**
Set up Python env. Build a synthetic organizational log generator
(login/logoff, file access, USB device usage, email metadata, HTTP
browsing) for a simulated 100-user organization over 30 days. Inject
labeled red-team insider scenarios (after-hours login, mass file access,
USB exfiltration, data exfiltration via email, privilege probing) with a
ground-truth `labels.csv` kept aside for evaluation only. Run a sanity-
check EDA to confirm malicious users stand out on simple aggregates.

**Day 2 — Deeper EDA + dataset calibration**
Expand exploratory analysis: per-user/day activity distributions, time-
of-day heatmaps, correlation between log types. Tune generator parameters
if the malicious/normal separation is too easy or too subtle. Optionally
layer in the CERT Insider Threat Dataset (r4.2) for cross-validation.

**Day 3 — Feature engineering**
Aggregate raw events into per-user/per-day behavioral features: login
time deviation from personal baseline, failed login counts, file access
volume/variance, USB event frequency, email send patterns (volume, size,
external ratio), off-hours activity ratio.

**Day 4 — Baseline unsupervised models**
Train Isolation Forest and One-Class SVM on the feature matrix. Evaluate
against the injected ground-truth labels (precision/recall/F1, ROC-AUC).

**Day 5 — Autoencoder + graph analysis**
Build an Autoencoder neural network (reconstruction-error anomaly
scoring) and compare against Day 4 baselines. Build a NetworkX graph of
user-resource-system interactions; compute degree/centrality to flag
unusual relationships and privilege misuse patterns.

**Day 6 — Explainable AI**
Add SHAP (tree-based models) and LIME (model-agnostic, local) so each
flagged alert comes with a human-readable explanation of why the user
was flagged.

**Day 7 — Streamlit dashboard + integration**
Build the SOC-style dashboard: alert list with anomaly scores, risk
indicators, the interaction graph, and the SHAP/LIME explanation panel
per alert. Run the full red-team test set through the pipeline end-to-
end and document results for the final report.
