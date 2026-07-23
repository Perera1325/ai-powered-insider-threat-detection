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
- [x] Day 5 — Autoencoder (from scratch) + graph analysis (NetworkX)
- [x] Day 6 — Explainable AI (from-scratch SHAP + LIME)
- [x] Day 7 — Streamlit dashboard + final integration

Daily write-ups: [Day 1](docs/day1_progress.md) · [Day 2](docs/day2_progress.md) ·
[Day 3](docs/day3_progress.md) · [Day 4](docs/day4_progress.md) · [Day 5](docs/day5_progress.md) ·
[Day 6](docs/day6_progress.md) · [Day 7](docs/day7_progress.md).
Generated figures live under [docs/figures/](docs/figures/).

**Read the Day 4-7 write-ups even if you only skim this repo** - each
documents an honest limitation or a real debugging finding (why the
"perfect" model scores aren't a real-world accuracy claim, why graph
analysis came up empty, why LIME was noisier than SHAP, why the
dashboard is verified via a stub rather than a live run) rather than
just reporting clean success. That's deliberate and worth carrying into
the final report.

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
  train_autoencoder.py     # Day 5: trains + evaluates the autoencoder
  graph_analysis.py        # Day 5: NetworkX user-resource graph + centrality
  models/
    isolation_forest.py    # Day 4: from-scratch Isolation Forest (see note in docs/day4_progress.md)
    mahalanobis.py          # Day 4: Mahalanobis detector (One-Class SVM stand-in, see same note)
    autoencoder.py           # Day 5: from-scratch Autoencoder (see note in docs/day5_progress.md)
  explain_day6.py            # Day 6: explains flagged users with SHAP + LIME
  explain/
    kernel_shap.py            # Day 6: from-scratch Kernel SHAP (see note in docs/day6_progress.md)
    lime_explainer.py          # Day 6: from-scratch LIME (see same note)
  dashboard_utils.py           # Day 7: testable core logic behind the dashboard
  live_demo.py                  # Live-demo mode: real-time simulated feed + injection
app.py                         # Day 7: Streamlit SOC dashboard (streamlit run app.py)
tests/
  _streamlit_stub.py           # Day 7: minimal streamlit stand-in (see docs/day7_progress.md)
  smoke_test_dashboard.py      # Day 7: runs app.py end-to-end against the stub (incl. Live Demo tab)
data/
  raw/                  # place manually-downloaded external datasets here (gitignored)
  processed/full/       # generated dataset (users, logon, device, http, email, file, labels)
  sample/               # small committed samples for quick inspection
docs/
  project_plan.md         # full 7-day build plan
  day1_progress.md ... day7_progress.md   # daily write-ups
  day1_eda_summary.md, day2_eda_summary.md, day3_ranking_check.md,
  day5_graph_summary.md, day6_explainability_report.md              # generated findings
  figures/                 # all generated plots
requirements.txt
```

## Training with the real libraries (Google Colab)

Everything in this repo's Day 4-6 work was implemented from scratch in
NumPy because the original dev sandbox could not install scikit-learn/
TensorFlow/shap/lime (documented in `docs/day4_progress.md` onward). If
you have access to Google Colab (or any machine with normal internet),
`notebooks/real_library_training.ipynb` re-runs the same evaluation
using the actual named libraries - scikit-learn's `IsolationForest` and
`OneClassSVM`, a TensorFlow/Keras Autoencoder, and real `shap`/`lime` -
against the exact same dataset already committed here, and compares the
results to the from-scratch versions.

**To run it:** open
`https://colab.research.google.com/github/Perera1325/ai-powered-insider-threat-detection/blob/main/notebooks/real_library_training.ipynb`
directly in your browser (no download/upload needed - Colab pulls it
straight from GitHub), then `Runtime > Run all`.

## Live demo mode

The dashboard has a **🔴 Live Demo** tab that simulates a small company
generating fresh activity day by day in real time, then injects an
insider attack mid-stream so you can watch the risk score spike as it
happens - built for presenting to a lecturer/audience rather than only
showing a static result. See
[docs/live_demo_guide.md](docs/live_demo_guide.md) for how to run and
narrate it.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# rebuild the full pipeline, in order:
python3 src/generate_logs.py
python3 src/inject_anomalies.py
python3 src/eda_day1.py
python3 src/eda_day2.py
python3 src/feature_engineering.py
python3 src/day3_ranking_check.py
python3 src/train_baseline_models.py
python3 src/train_autoencoder.py
python3 src/graph_analysis.py
python3 src/explain_day6.py

# smoke-test the dashboard logic without needing streamlit installed:
python3 tests/smoke_test_dashboard.py

# run the actual dashboard (needs `streamlit` + `pyarrow`, see note below):
streamlit run app.py
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

## Note on the dashboard (Day 7)

`app.py` is a full Streamlit dashboard, but this project's dev sandbox
could not install Streamlit's `pyarrow` dependency (same bandwidth
issue as scipy/TensorFlow in Day 4/5). The dashboard's actual logic
(`src/dashboard_utils.py`) was tested directly and via a stub-based
smoke test (`tests/smoke_test_dashboard.py`) that runs `app.py`
end-to-end without needing Streamlit installed - but nobody has seen
it actually render yet. On a normal connection, `pip install streamlit`
takes seconds; run `streamlit run app.py` and it should just work.
Please try it and let me know if anything looks off - see
[docs/day7_progress.md](docs/day7_progress.md) for full details.

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
