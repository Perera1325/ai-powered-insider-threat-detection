# Day 7 Progress - Dashboard + Final Integration

## What was built

- `app.py` — the full Streamlit SOC dashboard, three tabs:
  - **Alerts**: KPI row (users monitored, alerts shown, high-risk count,
    ground-truth matches), a ranked alert table with a 0-100 risk score
    and High/Medium/Low risk level, a model selector (Isolation Forest /
    Autoencoder / Mahalanobis detector), and a per-user drill-down with
    live SHAP + LIME explanations (Day 6) and that user's graph
    neighbourhood (Day 5).
  - **Interaction Graph**: the full user-resource graph from Day 5.
  - **Model Comparison**: the Day 4/5 comparison table and ROC curves.
- `src/dashboard_utils.py` — every piece of actual logic (data loading,
  model training/scoring, risk-level bucketing, SHAP/LIME explanation,
  graph building) as plain, independently-callable functions. `app.py`
  itself is a thin layer of `st.*` calls wired to these.
- `tests/_streamlit_stub.py` + `tests/smoke_test_dashboard.py` — see
  below for why this exists.

## Why a stub instead of running the real dashboard

`streamlit` itself downloads fine (~10MB in ~15s in this sandbox), but
it hard-depends on `pyarrow`, which - like `scipy` in Day 4 and
`tensorflow` in Day 5 - is too large to fetch over this sandbox's
throttled outbound proxy. So the real `streamlit run app.py` could not
be executed or screenshotted here.

To still verify correctness, the app was split into two layers:

1. `src/dashboard_utils.py` — all real logic, tested directly with
   plain Python (loading data, training all 3 models, building the
   alert table, computing SHAP/LIME for a live user, building the
   interaction graph and a per-user subgraph) - every function was
   run and its output checked by hand before being wired into the UI.
2. `tests/smoke_test_dashboard.py` — a stub `streamlit` module
   (`tests/_streamlit_stub.py`) that implements just enough of the
   Streamlit API (widgets, layout, caching decorators, `st.image` /
   `st.pyplot` / `st.dataframe`) to let `app.py` run top-to-bottom via
   `runpy`, catching real bugs (wrong file paths, bad function
   signatures, KeyErrors) without needing Streamlit itself installed.
   **This does not verify the visual layout** - only that the script
   executes without errors.

**To see the actual dashboard:** clone the repo on a machine with
normal internet access, `pip install -r requirements.txt` (streamlit +
pyarrow install in seconds outside this sandbox), then:

```bash
streamlit run app.py
```

## Verification performed

- `python3 -m py_compile app.py` — syntax check, passed
- Direct calls to every `dashboard_utils.py` function with the real
  Day 1-6 data — all returned sane shapes/values (verified: alert
  table ranks the 6 malicious users at the top with the same
  Isolation Forest scores as Day 4/6; SHAP output for U0027 matches
  Day 6's report; graph has 104 nodes/5097 edges, matching Day 5)
- `python3 tests/smoke_test_dashboard.py` — full script execution via
  the stub, no exceptions

## Project status: all 7 days complete

The end-to-end pipeline (synthetic logs → red-team labels → engineered
features → 4 anomaly-detection models → graph analysis → SHAP/LIME
explanations → dashboard) is implemented and committed. See the main
[README.md](../README.md) for the repo structure and the honest caveats
collected across each day (Day 4's "perfect scores" caveat, Day 5's
graph-analysis null result, Day 6's LIME-vs-SHAP finding) - these are
worth keeping front-and-center in the final report as evidence of
genuine critical evaluation, not just a working demo.

## Suggested next steps (beyond the 7-day sprint)

1. Apply the Day 5 fix recommendation (department-scoped file/email
   access) to make graph analysis actually informative, then re-run
   Day 5-7.
2. Add "grey area" borderline-normal users to the dataset so Day 4's
   evaluation isn't a clean 100% - a more defensible number for the
   final report.
3. Swap the from-scratch models for real scikit-learn / TensorFlow /
   shap / lime once running with normal internet access, and re-run
   the comparison to confirm the from-scratch versions tracked them
   closely.
4. Run `streamlit run app.py` locally and take real screenshots for
   the final report's dashboard section.
