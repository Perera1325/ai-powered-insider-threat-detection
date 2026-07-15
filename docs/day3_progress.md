# Day 3 Progress

## What was built

- `src/feature_engineering.py` — aggregates the five raw logs into a
  25-feature per-user table covering the whole 30-day window:
  - **Logon**: total logons, avg/std logon hour, after-hours logon
    count/ratio, avg/std session duration
  - **Device**: connect count, after-hours connect count
  - **HTTP**: event count, distinct domains visited
  - **Email**: send count, avg size, total attachments, external count/
    ratio, max external attachment size
  - **File**: event count, distinct files, copy/delete counts and
    ratio, max single-day event burst, shared-folder-category
    diversity, max single-day shared-folder burst
  - Output: `data/processed/full/user_features.csv` (label-free — this
    is what Day 4's models will actually train on) and
    `user_features_preview_with_labels.csv` (labels joined in, for our
    own inspection only)
- `src/day3_ranking_check.py` — a quick, non-ML sanity check: standardize
  every feature, sum absolute z-scores per user, and see where the 6
  known malicious users land in that ranking.

## Key finding

**All 6 malicious users rank 1st through 6th out of 100** on the
combined z-score, clearly ahead of the highest-scoring normal user
(rank 7). This is a big improvement over Day 2, where 3 of the 6
scenarios were indistinguishable from normal variance on raw counts
alone — the engineered features (after-hours ratios, session-duration
stats, burst detection, folder-category diversity) capture the
behavioural signal that raw totals missed.

This is a naive linear combination, not a trained model, so it likely
overstates how easy the problem is in general (with only 6 labeled
positives, it's easy for any reasonable feature set to separate them).
Day 4's Isolation Forest / One-Class SVM will be the first real,
properly-evaluated (precision/recall/F1) test of this feature set.

## Next (Day 4)

Train Isolation Forest and One-Class SVM on `user_features.csv`
(without labels), then evaluate predictions against `labels.csv`.
