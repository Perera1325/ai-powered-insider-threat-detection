# AI-Powered Insider Threat Detection

[![tests](https://github.com/Perera1325/ai-powered-insider-threat-detection/actions/workflows/tests.yml/badge.svg)](https://github.com/Perera1325/ai-powered-insider-threat-detection/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.12-blue)
![streamlit](https://img.shields.io/badge/streamlit-1.67%2B-1D4ED8)

An unsupervised insider-threat detection system with a SOC-style dashboard. It learns what normal
behaviour looks like for an organisation, flags users who deviate from it, explains **why** each user was
flagged (SHAP and LIME), shows their network context, and lets an analyst work the alerts as cases.
Final year project (BSc (Hons) Cyber Security Top-up).

**Student:** T. Siyath Sathnidu (HCL/IT/CYBR/2097)

> All data is synthetic. No real employee, customer or organisational data is used, collected or stored.

---

## Quick start (about 2 minutes)

The generated dataset is committed to the repository, so you do **not** need to build anything first.

```bash
git clone https://github.com/Perera1325/ai-powered-insider-threat-detection.git
cd ai-powered-insider-threat-detection

python3 -m venv .venv
source .venv/bin/activate            # Git Bash on Windows: source .venv/Scripts/activate
pip install -r requirements.txt

streamlit run app.py
```

Streamlit prints a local address (normally <http://localhost:8501>); open it in a browser.

| You want | Run |
|---|---|
| The dashboard (lightweight, no TensorFlow) | `pip install -r requirements.txt` then `streamlit run app.py` |
| Everything, including the Keras autoencoder and the test runner | `pip install -r requirements-dev.txt` |
| The test suite | `python -m pytest -q tests` (about 2 minutes) |
| Rebuild the dataset from scratch | see [Rebuilding the pipeline](#rebuilding-the-pipeline) |

Python 3.12 is recommended (it is what CI and the hosted app use).

## Using the dashboard

The sidebar sets the **detection model** and how many alerts to show. Every tab follows that model.

| Tab | What it does |
|---|---|
| **Alert queue** | Ranked risk table. Click a row to open that user, read the SHAP / LIME explanation and the user's resource neighbourhood, set a **case status** (New, Investigating, Escalated, Closed) with an analyst note, and export the queue as **CSV or PDF**. Cases are **saved to a SQLite database** (they survive a page refresh) and can also be exported/restored as CSV. Each alert shows suggested **MITRE ATT&CK** techniques. |
| **Investigate** | One page per entity: daily activity timeline, most unusual behaviours (z-scores), comparison with the user's department, model explanation, network neighbourhood and the case record. |
| **Real-time monitor** | A separate process streams log events; the scorer tails them, rebuilds each user's features over a 30-day sliding window and raises alerts. Live KPIs, risk leaderboard, heat map, pipeline latency, alert log with top drivers and an event feed. Pause, resume, change the speed or the threshold while it runs, and inject an attack on demand. |
| **Attack simulation** | Scripted demo: pick a user and a scenario, start the feed and watch that user's risk jump the moment the attack is injected. |
| **Entity graph** | Deterministic (non-animated) organisation graph coloured by risk, with network statistics and a focus mode for one user and their contacts. |
| **Model evaluation** | Do the models agree (top-k overlap and consensus alerts)? **Threshold simulator** (precision, recall, false alarms and analyst workload as you move the threshold), plus the Day 4 and multi-seed evaluation results. |
| **Upload & analyse** | Bring your own data: upload logon / device / http / email / file logs (and optionally users) or a ready-made feature table, then score it with the selected model. |
| **MITRE ATT&CK** | Maps each alert's strongest behavioural deviations to ATT&CK techniques (heuristic investigation hints): coverage per technique, one user's evidence, and a validation check against the synthetic answer key. |
| **Notifications & audit** | Feed of real-time alerts and escalated cases (optionally also pushed to a Slack-style webhook). Administrators also see the audit log: sign-ins, case changes, exports and stream control. |

### Sign-in and roles

The dashboard asks for a login. Demo accounts (shown on the login page): `admin / admin123`, `analyst / analyst123`, `viewer / viewer123`.

| Role | Can do |
|---|---|
| viewer | Read everything except the audit log; cannot change cases, export, or control the stream |
| analyst | Everything a viewer can, plus case updates, exports, stream and attack controls |
| admin | Everything, plus the audit log, restoring a case log and test notifications |

For a real deployment set your own accounts in `.streamlit/secrets.toml` (`[users.alice]` with `password` or a `password_hash` from `auth.hash_password`, and `role`), set `webhook_url` for push alerts, and set `ITD_SHOW_DEMO_LOGINS=off`. `ITD_AUTH=off` disables the login for local development. The demo accounts are for demonstration only.

### Try this in a demo (3 minutes)

1. **Alert queue** - click the top row, read the SHAP explanation, set it to *Escalated* with a note, download the PDF.
2. **Investigate** - open the same user and show the timeline spike and the z-score chart.
3. **Real-time monitor** - *Start stream*, wait a few ticks, *Inject attack now*, watch the risk line cross the threshold and the alert appear; drag the threshold slider.
4. **Model evaluation** - show the agreement heat map, then the threshold simulator trade-off.

[docs/live_demo_guide.md](docs/live_demo_guide.md) has a longer script.

### Upload your own logs

On the **Upload & analyse** tab, choose either

- the raw logs (any of `logon.csv`, `device.csv`, `http.csv`, `email.csv`, `file.csv`, plus optional `users.csv`), or
- a feature table with a `user_id` column and the 28 feature columns.

The app checks the schema, builds per-user features, scores everyone with the selected model and lists the most
unusual users with their top drivers. At least 20 users are required. Examples of the expected columns are in
`data/processed/full/`.

## How it works

```
 synthetic logs ──► red-team injection ──► feature engineering ──► anomaly models ──► risk 0-100
 (logon, device,    (labels kept apart,     (28 behavioural        Isolation Forest      │
  http, email,       evaluation only)        features per user)    One-Class SVM         ▼
  file)                                                            Autoencoder     explanations (SHAP, LIME)
                                                                   Mahalanobis      graph context (NetworkX)
                                                                                    case management + export
```

1. **Log simulation** - synthetic organisation activity (logon, device/USB, HTTP, e-mail, file access).
2. **Red-team injection** - labelled insider scenarios added to the baseline; labels are used only to evaluate.
3. **Feature engineering** - 28 behavioural and statistical features per user.
4. **Anomaly detection** - scikit-learn Isolation Forest and One-Class SVM, a Keras autoencoder, a Mahalanobis detector, and from-scratch versions kept as cross-checks. Scores are mapped to a 0-100 risk scale.
5. **Explainability** - the `shap` KernelExplainer and `lime` tabular explainer.
6. **Graph analysis** - user / folder / e-mail relationships with NetworkX (degree, betweenness, modularity).
7. **Real-time pipeline** - micro-batch producer and scorer, absolute risk scale learned from the training population.
8. **Analyst workflow** - case management, investigation view, threshold tuning and report export.

Risk is an **unsupervised anomaly score**: it indicates behaviour that deviates from the population, not proof of
malicious intent.

## Results

A single 100-user dataset scores close to 1.0 for most models, which is **not** a quality claim. The robust
evaluation uses 8 unseen datasets (200 users, 10 insiders each) and then weakens the attack:

| Model | ROC-AUC (mean ± std) | precision@k |
|---|---|---|
| Mahalanobis (scratch) | 0.998 ± 0.002 | 0.94 |
| Autoencoder (Keras, winsorised) | 0.976 ± 0.023 | 0.85 |
| Isolation Forest (sklearn) | 0.974 ± 0.024 | 0.83 |
| Z-score baseline | 0.954 ± 0.027 | 0.84 |
| Isolation Forest (scratch) | 0.953 ± 0.039 | 0.66 |
| One-Class SVM (sklearn) | 0.938 ± 0.021 | 0.83 |
| Autoencoder (scratch) | 0.805 ± 0.123 | 0.59 |

Details, the low-and-slow stress test and caveats: [docs/robust_evaluation.md](docs/robust_evaluation.md).

## Limitations (read these)

- **Saved cases** live in a local SQLite file (`data/runtime/itd.db`). On Streamlit Community Cloud the disk is reset when the app is rebuilt, so export the case log CSV for a durable copy.
- **Login** is a demo-grade gate (hashed passwords, roles, session lockout); it is not a replacement for single sign-on.
- **ATT&CK mapping** is heuristic (feature to technique) and only suggests where to look.

- **Synthetic data.** Results show the method works on this generator; they are not a claim about real organisations.
- **Ground truth is used only for evaluation.** The *Threshold simulator* and the "ground truth" columns use the dataset's answer key. A production system would not have them, so treat the simulator as a tuning aid for the demo data.
- **Case records live in the browser session.** Use *Download case log* / *Restore a saved case log* to keep them; hosted apps forget them on restart.
- **Hosted app has no TensorFlow.** The free hosting tier cannot hold it, so the Keras autoencoder appears only in a local install of `requirements-dev.txt`.
- **A Keras autoencoder trained for many epochs on about 100 users memorises the insiders**, and scikit-learn's default One-Class SVM kernel is too narrow for 28 features. Both findings are documented in [docs/robust_evaluation.md](docs/robust_evaluation.md).
- **Graph position does not predict risk** (Spearman rho near 0 on this data): the graph is context for an analyst, not a detector. See [docs/day5_progress.md](docs/day5_progress.md).

## Rebuilding the pipeline

Only needed if you want to regenerate the dataset or the figures. Run from the repository root, in this order:

```bash
python3 src/generate_logs.py
python3 src/inject_anomalies.py
python3 src/feature_engineering.py
python3 src/train_baseline_models.py
python3 src/train_autoencoder.py        # needs tensorflow-cpu (requirements-dev.txt)
python3 src/graph_analysis.py
python3 src/explain_day6.py
python3 src/pyvis_graph.py              # interactive graph HTML
python3 src/robust_evaluation.py        # multi-seed evaluation (about 3 minutes)
```

`src/eda_day1.py`, `src/eda_day2.py` and `src/day3_ranking_check.py` are optional exploratory reports.

## Tests and CI

```bash
python -m pytest -q tests
```

70 tests cover data generation, features, models, explainers, the graph, the real-time pipeline (a real producer
subprocess), the analyst-workflow helpers, and the real Streamlit app driven headlessly with Streamlit's `AppTest`.
GitHub Actions runs the suite on every push ([.github/workflows/tests.yml](.github/workflows/tests.yml)).
Test report: [docs/final_test_report.md](docs/final_test_report.md).

## Deployment

The app is deployed on Streamlit Community Cloud from `main` with `app.py` as the entry point. Steps, platform
comparison and known limits: [docs/deployment.md](docs/deployment.md). Choose **Python 3.12** under *Advanced
settings*; the dependencies come from `requirements.txt`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit: command not found` | Activate the virtual environment first, or run `python -m streamlit run app.py`. |
| `ModuleNotFoundError` after a `git pull` | Dependencies changed: run `pip install -r requirements.txt` again. |
| Autoencoder (Keras) missing from the model list | Install TensorFlow: `pip install -r requirements-dev.txt`. |
| PDF export button is disabled | `pip install reportlab`. |
| Port 8501 already in use | `streamlit run app.py --server.port 8502`. |
| Real-time tab shows no data | Press *Start stream*; the first tick takes a moment while the models score. |
| Hosted app is asleep | Open it and wait 30-60 seconds; free apps sleep after 12 hours without visitors. |

## Repository layout

```
app.py                       Streamlit dashboard (9 tabs)
requirements.txt             runtime dependencies (used by the hosted app)
requirements-dev.txt         + TensorFlow (Keras autoencoder) and pytest
src/
  generate_logs.py           synthetic log generator
  inject_anomalies.py        red-team scenarios + ground-truth labels
  feature_engineering.py     per-user feature table (28 features)
  train_baseline_models.py   baseline model training and comparison
  train_autoencoder.py       autoencoder training and evaluation
  robust_evaluation.py       8 unseen seeds + attack-strength stress test
  graph_analysis.py          NetworkX graph and centrality
  models/                    sklearn / Keras detectors plus from-scratch versions
  explain/                   shap + lime wrappers plus from-scratch versions
  realtime/                  event_producer.py (log stream) and scorer.py (sliding-window scoring)
  dashboard_utils.py         testable core logic behind the dashboard
  soc_tools.py               case management, drill-down, threshold sweep, model agreement, CSV/PDF export
  mitre.py                   MITRE ATT&CK mapping (scenarios, features, coverage, validation)
  case_store.py              SQLite store for cases, audit log and notifications
  auth.py                    PBKDF2 password hashing, roles and permissions
  notify.py                  alert notifications (feed + optional https webhook)
  network_view.py            deterministic network graph and statistics
  upload_analysis.py         schema checks and features for user-uploaded data
  charts.py, ui_theme.py     blue/white chart and page styling
  live_demo.py               scripted attack-simulation feed
tests/                       pytest suite (70 tests)
data/processed/full/         generated dataset (committed); data/raw/ for optional external data
docs/                        daily write-ups, evaluation, deployment, figures
notebooks/                   optional Colab notebook for the real-library training
```

Daily write-ups: [Day 1](docs/day1_progress.md) · [Day 2](docs/day2_progress.md) · [Day 3](docs/day3_progress.md) ·
[Day 4](docs/day4_progress.md) · [Day 5](docs/day5_progress.md) · [Day 6](docs/day6_progress.md) ·
[Day 7](docs/day7_progress.md) · [Day 8](docs/day8_progress.md). Real-time design:
[docs/realtime.md](docs/realtime.md). Build plan: [docs/project_plan.md](docs/project_plan.md).

## About the CERT dataset

The proposal lists the CERT Insider Threat Dataset (r4.2, CMU/DARPA) as an optional extra source. It is several GB
and is not bundled. To use it, download it manually from CMU's Figshare repository or Kaggle, place the raw files
under `data/raw/` (git-ignored) and load it with `src/load_cert.py`.

## Ethics

All data is synthetically generated or taken from public research datasets such as CERT. No real personal or
organisational data is processed. The system is built for academic and defensive cybersecurity research and is not
intended for deployment in a live environment.
