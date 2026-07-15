# AI-Powered Insider Threat Detection System

Final Year Project (BSc (Hons) Cyber Security Top-up) — an AI-driven
insider threat detection system combining unsupervised machine learning,
graph analysis, and explainable AI (XAI) to identify anomalous user
behaviour within simulated organizational environments.

**Student:** T. Siyath Sathnidu (HCL/IT/CYBR/2097)

This project does not use, collect, or store any real personal or
confidential organizational data. All activity logs are synthetically
generated for research and academic purposes only.

## Project pipeline

1. **Log simulation** — synthetic organizational activity logs (logon,
   device/USB, HTTP, email, file access)
2. **Red-team injection** — labeled insider-threat scenarios injected
   into the baseline logs, kept separate as ground truth for evaluation only
3. **Feature engineering** — behavioural, statistical, and graph features
4. **Anomaly detection** — Isolation Forest, One-Class SVM, Autoencoder
5. **Graph analysis** — user/system/resource relationship graphs (NetworkX)
6. **Explainable AI** — SHAP and LIME explanations for flagged alerts
7. **Dashboard** — interactive Streamlit SOC-style dashboard

See [docs/project_plan.md](docs/project_plan.md) for the full day-by-day
build plan.

## Status

- [x] Day 1 — synthetic dataset generator + red-team anomaly injection + EDA
- [x] Day 2 — deeper EDA / dataset calibration (heatmaps, correlations, CERT loader stub)
- [x] Day 3 — feature engineering (25 behavioral/statistical features per user)
- [x] Day 4 — baseline unsupervised models (Isolation Forest + Mahalanobis detector; see note below on One-Class SVM)
- [ ] Day 5 — Autoencoder + graph analysis
- [ ] Day 6 — Explainable AI (SHAP, LIME)
- [ ] Day 7 — Streamlit dashboard + final integration

See [docs/day1_progress.md](docs/day1_progress.md) / [docs/day2_progress.md](docs/day2_progress.md)
for daily write-ups, and [docs/day1_eda_summary.md](docs/day1_eda_summary.md) /
[docs/day2_eda_summary.md](docs/day2_eda_summary.md) for the EDA findings.
Generated figures live under [docs/figures/](docs/figures/).

## Repository structure

```
src/
  generate_logs.py         # synthetic log generator (Day 1)
  inject_anomalies.py      # red-team scenario injection + ground truth labels
  eda_day1.py              # Day 1 sanity-check EDA
  eda_day2.py              # Day 2 deeper EDA (heatmaps, correlations, comparisons)
  load_cert.py             # optional CERT r4.2 loader (for later cross-validation)
  feature_engineering.py   # Day 3: builds per-user behavioral feature table
  day3_ranking_check.py    # Day 3: naive z-score ranking sanity check
  train_baseline_models.py # Day 4: trains + evaluates unsupervised models
  models/
    isolation_forest.py    # Day 4: from-scratch Isolation Forest (see note in docs/day4_progress.md)
    mahalanobis.py          # Day 4: Mahalanobis detector (One-Class SVM stand-in, see same note)
data/
  raw/                  # place manually-downloaded external datasets here (gitignored)
  processed/full/       # generated dataset (users, logon, device, http, email, file, labels)
  sample/               # small committed samples for quick inspection
docs/
  project_plan.md       # full 7-day build plan
  day1_progress.md      # Day 1 write-up
  day1_eda_summary.md    # auto-generated EDA output
requirements.txt
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 src/generate_logs.py
python3 src/inject_anomalies.py
python3 src/eda_day1.py
```

## Note on Day 4 models (Isolation Forest / One-Class SVM)

Day 4 was built and evaluated in a sandbox whose network proxy could
not download scikit-learn's scipy dependency in reasonable time
(verified ~40-50 KB/s on the wheel). Isolation Forest was implemented
from scratch (pure NumPy, see `src/models/isolation_forest.py`); a
Mahalanobis-distance detector stands in for One-Class SVM (see
`src/models/mahalanobis.py`). On a normal connection,
`pip install scikit-learn` is instant - swap in
`sklearn.ensemble.IsolationForest` / `sklearn.svm.OneClassSVM` there if
you want the exact algorithms named in the proposal (same `.fit()` /
`.decision_function()` interface, drop-in replacement). Full reasoning
and results are in [docs/day4_progress.md](docs/day4_progress.md).

## Note on the CERT Insider Threat Dataset

The project proposal lists the CERT Insider Threat Dataset (r4.2, CMU/DARPA)
as an optional supplementary data source. It was not bundled into this
repo because it is several GB and not suitable for version control. If
you want to use it: download it manually from CMU's Figshare
repository ("Insider Threat Test Dataset") or Kaggle, place the raw
files under `data/raw/`, and it can be loaded alongside the synthetic
dataset for cross-validation later in the project.

## Ethics

All data used in this project is synthetically generated or drawn from
publicly available research datasets (e.g. CERT). No real employee,
customer, or organizational data is collected or processed. The system
is developed strictly for academic and defensive cybersecurity research
and is not intended for deployment in a live environment.
