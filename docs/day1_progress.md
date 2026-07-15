# Day 1 Progress

## What was built

- `src/generate_logs.py` — generates a synthetic organizational
  directory (100 users across 7 departments/4 roles) and five baseline
  activity logs over a 30-day period (weekdays only): logon, device
  (USB), HTTP browsing, email metadata, and file access. Timestamps
  follow realistic business-hour distributions.
- `src/inject_anomalies.py` — selects 6 users and injects one labeled
  red-team scenario each: `after_hours_login`, `mass_file_access`,
  `usb_exfiltration`, `data_exfil_email`, `privilege_probe`. Ground
  truth is written to `data/processed/full/labels.csv` and is used only
  for evaluation, never as a model input.
- `src/eda_day1.py` — sanity-check EDA confirming the injected users are
  distinguishable from the normal population on simple aggregates
  (see `docs/day1_eda_summary.md`).

## Dataset generated

| File        | Rows   |
|-------------|-------:|
| users.csv   | 100    |
| logon.csv   | ~4,200 |
| device.csv  | ~270   |
| http.csv    | ~13,400|
| email.csv   | ~6,600 |
| file.csv    | ~17,600|

Total on-disk size: ~3 MB — small enough to commit directly to the repo
under `data/processed/full/`.

## Sanity-check findings

- The two `after_hours_login` users are the *only* users with logon
  events outside 06:00-20:00 — a clean, false-positive-free signal.
- The `data_exfil_email` user sends a ~22 MB external attachment vs. a
  ~600 KB max for everyone else — an obvious outlier.
- `mass_file_access` / `privilege_probe` users land at the top of the
  file-event-count ranking, though not by an extreme margin — this is
  expected for a subtler scenario and something to refine further with
  proper feature engineering (Day 3) rather than raw counts.

## Decision on data source

Per the project proposal, the CERT Insider Threat Dataset (r4.2) was
considered as a supplementary/optional source. It was not pulled into
this environment because (a) it's several GB, unsuitable for direct
version control, and (b) the network available while building this
repo could not reach CMU's Figshare/Kaggle hosting. The synthetic
generator above satisfies Objective 1 ("generate simulated log files
reflecting organizational activities") independently. CERT can still be
downloaded manually later and cross-referenced (see README).

## Next (Day 2)

Deeper EDA (time-of-day heatmaps, per-log-type correlations) and, if
needed, calibrate generator parameters before moving into feature
engineering.
