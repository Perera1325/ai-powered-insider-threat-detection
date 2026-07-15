# Day 4 Progress

## What was built

- `src/models/isolation_forest.py` — a from-scratch Isolation Forest
  (Liu, Ting & Zhou, 2008), pure NumPy.
- `src/models/mahalanobis.py` — a Mahalanobis-distance multivariate
  anomaly detector, pure NumPy, used as a stand-in for One-Class SVM.
- `src/train_baseline_models.py` — trains both on the label-free
  `user_features.csv`, scores every user, and evaluates the rankings
  against `labels.csv` (precision/recall/F1 at top-K, ROC-AUC — all
  computed manually, no sklearn.metrics).
- `docs/figures/day4_roc_curves.png` — ROC curves for all three
  approaches (Isolation Forest, Mahalanobis, Day 3's z-score baseline).

### Why hand-rolled models instead of scikit-learn

The proposal specifies Isolation Forest and One-Class SVM via
scikit-learn. This dev sandbox's outbound network proxy throttles
uncached large binary wheels heavily — a direct `curl` timing test
showed roughly 40–50 KB/s to `files.pythonhosted.org` for scipy
(scikit-learn's core dependency, ~35–40 MB), which made `pip install
scikit-learn` impossible to complete inside this environment. Rather
than block on that, Isolation Forest was implemented directly (the
algorithm is simple enough to reproduce faithfully) and a Mahalanobis
distance detector was used in place of One-Class SVM (same role —
learn a "normal" region from unlabeled data, score deviations from it
— but a linear/statistical boundary rather than an SVM's learned
non-linear margin).

**On a normal internet connection this is a non-issue** — `pip install
scikit-learn` completes in seconds. Recommended next step when running
locally: install scikit-learn and swap in
`sklearn.ensemble.IsolationForest` / `sklearn.svm.OneClassSVM(kernel="rbf")`
to get the exact models the proposal names; the current code is
structured so that's a drop-in replacement (same `.fit()` /
`.decision_function()` interface).

## Results

| Model                     | Precision@6 | Recall@6 | F1@6 | ROC-AUC |
|---------------------------|:-----------:|:--------:|:----:|:-------:|
| Day 3 z-score baseline    | 1.0         | 1.0      | 1.0  | 1.0     |
| Isolation Forest (scratch)| 1.0         | 1.0      | 1.0  | 1.0     |
| Mahalanobis detector      | 1.0         | 1.0      | 1.0  | 1.0     |

All three approaches perfectly separate the 6 known malicious users
from the 94 normal users (flagging the top 6 by anomaly score = all 6
true positives, zero false positives).

## Important caveat — read this before quoting "100%" anywhere

A perfect AUC on your own synthetic dataset is an expected result of
the evaluation setup, not evidence the detection problem is generally
solved:

1. **Small, clean-signal dataset.** 100 users, 6 labeled insiders, and
   scenarios injected with clearly abnormal magnitudes (e.g. a ~22MB
   external email attachment vs. a ~600KB normal maximum). Day 3's
   engineered features already made these 6 users rank 1st–6th on a
   naive combined z-score before any model was trained — the models
   here are confirming that signal, not discovering something subtle.
2. **No noisy/ambiguous "grey area" insiders.** Real environments have
   users whose behaviour is borderline suspicious without being
   malicious (e.g. legitimately working late, or a manager who
   genuinely needs cross-department file access) — none of that noise
   exists in this dataset yet.
3. **k is set using the known label count.** "Flag top-6" is only
   possible because we know there are 6 malicious users; a real
   deployment doesn't get this SEK information and has to pick a
   threshold or contamination rate blind.

**Recommendation for the final report:** present this result as
"pipeline validation" — proof the end-to-end system (log → features →
model → evaluation) works correctly — rather than as a claim about
real-world detection accuracy. A stronger, more defensible evaluation
(and a good thing to add before submission) would be adding harder,
more ambiguous "borderline normal" users to the dataset and re-running
this same evaluation to show a more realistic, non-perfect precision/
recall trade-off.

## Next (Day 5)

Autoencoder neural network (reconstruction-error scoring) — will also
need a from-scratch or lightweight approach if TensorFlow can't be
installed here for the same bandwidth reasons; and graph analysis with
NetworkX (need to check if that installs cleanly — much smaller
package than scipy).
