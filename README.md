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
- [x] Day 8 — real libraries wired in (scikit-learn IsolationForest/OneClassSVM, TensorFlow/Keras autoencoder, `shap`, `lime`), multi-seed robust evaluation, interactive PyVis graph, recipient-metadata email features, 59-test pytest suite, dashboard verified with real Streamlit

Daily write-ups: [Day 1](docs/day1_progress.md) · [Day 2](docs/day2_progress.md) ·
[Day 3](docs/day3_progress.md) · [Day 4](docs/day4_progress.md) · [Day 5](docs/day5_progress.md) ·
[Day 6](docs/day6_progress.md) · [Day 7](docs/day7_progress.md) · [Day 8](docs/day8_progress.md).
Final evaluation: [docs/robust_evaluation.md](docs/robust_evaluation.md) · test report: [docs/final_test_report.md](docs/final_test_report.md).
Generated figures live under [docs/figures/](docs/figures/).

**Read the Day 4-7 write-ups even if you only skim this repo** - each
documents an honest limitation or a real debugging finding (why the
"perfect" model scores aren't a real-world accuracy claim, why graph
analysis came up empty, why LIME was noisier than SHAP, why the
dashboard was first verified via a stub - superseded in Day 8 by real Streamlit tests) rather than
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
  models/real_models.py        # Day 8: scikit-learn IF / One-Class SVM + Keras Autoencoder (drop-in, same interface)
  explain/real_explainers.py   # Day 8: real `shap` KernelExplainer + `lime` tabular explainer
  pyvis_graph.py               # Day 8: interactive PyVis user-resource graph (HTML)
  realtime/event_producer.py   # Day 8: separate-process log stream (JSON lines), on-demand attack injection
  realtime/scorer.py           # Day 8: incremental sliding-window scorer + alerts (see docs/realtime.md)
  robust_evaluation.py         # Day 8: 8 unseen seeds + attack-strength stress test
  dashboard_utils.py           # Day 7: testable core logic behind the dashboard
  live_demo.py                  # Live-demo mode: real-time simulated feed + injection
app.py                         # Day 7: Streamlit SOC dashboard (streamlit run app.py)
tests/                          # Day 8: pytest suite (59 tests) incl. the real Streamlit app via AppTest
  test_data.py  test_pipeline.py  test_models.py  test_explain_graph.py
  test_live_demo.py  test_dashboard_app.py  test_performance.py  test_realtime.py  conftest.py
.github/workflows/tests.yml     # CI: runs the suite on every push
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

## Real libraries

Days 4-6 were first written from scratch in NumPy because the original dev
sandbox could not install scikit-learn / TensorFlow / shap / lime. Day 8 wires
the real libraries into the main code path (`src/models/real_models.py`,
`src/explain/real_explainers.py`) and the dashboard offers all of them; the
from-scratch versions are kept as cross-checks. The Colab notebook
(`notebooks/real_library_training.ipynb`) still works but is now optional.

## Results (honest numbers)

The single 100-user dataset scores ~1.0 for most models, which is not a quality
claim. `python3 src/robust_evaluation.py` evaluates on **8 unseen datasets**
(200 users, 10 insiders each) and then weakens the attack (alpha = 1 -> 0.25):

| Model | ROC-AUC (mean ± std) | precision@k |
|---|---|---|
| Mahalanobis (scratch) | 0.998 ± 0.002 | 0.94 |
| Autoencoder (Keras, winsorised) | 0.976 ± 0.023 | 0.85 |
| Isolation Forest (sklearn) | 0.974 ± 0.024 | 0.83 |
| Z-score baseline | 0.954 ± 0.027 | 0.84 |
| Isolation Forest (scratch) | 0.953 ± 0.039 | 0.66 |
| One-Class SVM (sklearn) | 0.938 ± 0.021 | 0.83 |
| Autoencoder (scratch) | 0.805 ± 0.123 | 0.59 |

Full tables, the low-and-slow stress test and caveats:
[docs/robust_evaluation.md](docs/robust_evaluation.md).

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
pip install -r requirements-dev.txt

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
python3 src/pyvis_graph.py          # interactive graph HTML
python3 src/robust_evaluation.py    # multi-seed evaluation (~3 min)

# run the whole test suite (~1.5 min):
python -m pytest -q tests

# run the dashboard:
streamlit run app.py
```

## Notes on the from-scratch models

`src/models/isolation_forest.py`, `mahalanobis.py` and `autoencoder.py` are the
Day 4-5 from-scratch implementations. They are kept for comparison. The
real-library versions are the defaults in the dashboard. Two findings worth
knowing (details in [docs/robust_evaluation.md](docs/robust_evaluation.md)):

- A Keras autoencoder trained for many epochs on ~100 users **memorises the
  insiders** (AUC fell from 0.95 to 0.38 as epochs went 10 -> 100); winsorised
  training (inputs clipped to ±3σ) fixes this.
- scikit-learn's default One-Class SVM `gamma="scale"` is too narrow for 28
  features (AUC 0.76 vs ~1.0 with a wider kernel).

## Dashboard

`streamlit run app.py` - tabs: Alerts (risk table + SHAP/LIME explanation per
alert), **Real-Time** (separate log-stream process + incremental scorer, micro-batch - see [docs/realtime.md](docs/realtime.md)), Live Demo (older scripted simulation), Interactive Graph (PyVis), Model
Comparison (incl. the robust evaluation). It is tested by `tests/test_dashboard_app.py`
using Streamlit's own `AppTest` runner against the real app.

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
