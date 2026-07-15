# Day 2 Progress

## What was built

- `src/eda_day2.py` — deeper EDA on the Day 1 dataset:
  - Hour-of-day x weekday heatmap of logon activity
  - Correlation matrix of per-user daily event counts across the five
    log types
  - Boxplots comparing malicious vs. normal users on file-event counts
    and max external email attachment size
- `src/load_cert.py` — optional loader that maps the CERT Insider
  Threat Dataset (r4.2)'s native schema onto this project's schema, for
  later cross-validation. Not run yet (dataset not present in
  `data/raw/` — network access to CMU/Kaggle was not available while
  building this repo; download is a manual step for later, see README).
- `docs/day2_eda_summary.md` — full write-up of findings
- `docs/figures/*.png` — the three generated plots

## Key findings

1. **Log types are largely independent.** No pairwise correlation
   between per-user daily counts across logon/device/http/email/file
   exceeds ~0.11. This confirms it's worth engineering features from
   each log type separately in Day 3 rather than collapsing them into
   one generic "activity volume" signal.
2. **Two of six injected scenarios are trivially separable even on raw
   counts** (`after_hours_login`, `data_exfil_email`) — this is a good
   sign the injection logic is correct end-to-end.
3. **The other three** (`mass_file_access`, `usb_exfiltration`,
   `privilege_probe`) overlap with normal-user variance on raw
   aggregates — malicious users are only ~20 file-events above the
   normal mean, well within the normal population's spread. This is
   intentional/expected: real insider threats rarely stand out on raw
   counts alone, which is exactly why Day 3's engineered features
   (after-hours ratio, per-department path deviation, burst detection)
   are needed rather than simple thresholds.

## Calibration decision

No changes to generator parameters. The current mix (2 "easy" + 3
"hard" scenarios out of 6 total) is a reasonable evaluation set: it
lets Day 4+ model evaluation show both near-100% recall cases and
genuinely challenging cases, rather than a dataset that's either
trivial or impossible.

## Next (Day 3)

Feature engineering: aggregate raw events into per-user/per-day
behavioral features (login time deviation, failed-login counts, file
access volume/variance, USB frequency, email patterns, off-hours
ratio) to make the three "hard" scenarios separable.
