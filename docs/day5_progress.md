# Day 5 Progress

## What was built

- `src/models/autoencoder.py` — from-scratch feedforward Autoencoder,
  pure NumPy (manual forward/backward pass, no TensorFlow — see the
  module docstring for why: TensorFlow is 200+ MB, far beyond what
  Day 4 already showed this sandbox's network can fetch).
- `src/train_autoencoder.py` — trains it on `user_features.csv`, scores
  by reconstruction error, and appends the result to the same
  `day4_model_comparison.csv` table from Day 4 (now covers 4 approaches).
- `src/graph_analysis.py` — builds a NetworkX graph from file-access
  (user → shared-folder-category) and internal-email (user ↔ user)
  activity, computes degree/weighted-degree/betweenness centrality and
  resource diversity per user.
- `docs/figures/day5_autoencoder.png` — training loss curve + ROC curve
- `docs/figures/user_resource_graph.png` — full graph visualization,
  malicious users highlighted in red

## Autoencoder: a real tuning finding, not just a result

The first configuration tried (12 hidden units, 6-dim bottleneck, 1500
training epochs) gave a poor result: **precision/recall@6 = 0, AUC =
0.63**. Investigating why: with enough capacity and training time, the
autoencoder learns to reconstruct *everyone* well — including the 6
malicious users — which defeats the purpose, since anomaly detection
here depends on the model generalizing "normal" behaviour well enough
that abnormal behaviour reconstructs poorly.

A small grid check (see commit) confirmed the pattern: larger
bottleneck + more epochs → lower reconstruction error overall but worse
separation; a **tighter bottleneck (2 dimensions) with fewer epochs
(200)** gave **precision/recall/F1/AUC = 1.0**, matching Day 4's other
models. This is a genuine, well-documented autoencoder anomaly-detection
lesson (capacity control / early stopping matters more than minimizing
raw training loss) and is worth keeping in the final report as evidence
of actual experimentation rather than a first-try success.

## Graph analysis: an honest null result

Unlike the behavioral features (Day 3/4) and the autoencoder above,
**graph centrality metrics did NOT separate malicious from normal
users** in this dataset:

| Group      | avg degree | avg weighted degree | avg betweenness | resource diversity |
|------------|:----------:|:--------------------:|:----------------:|:-------------------:|
| Normal     | 97.9       | 440.0                 | 0.0004           | 4.0                 |
| Malicious  | 97.8       | 457.5                 | 0.0004           | 4.0                 |

Two dataset design reasons, both worth fixing before the final report:

1. **The email graph is nearly complete.** `generate_logs.py` picks
   email recipients uniformly at random from all 100 users, so almost
   every user ends up connected to almost every other user within 30
   days — there's no sparse social structure for centrality to be
   informative against.
2. **Shared-folder access isn't department-scoped for normal users.**
   All users access all 4 shared folders (reports/hr/finance/projects)
   over the month regardless of their own department, so
   `privilege_probe` (accessing a foreign department's folder) doesn't
   look structurally unusual — every user already has
   `resource_diversity = 4`.

**Recommended fix for a stronger final-report result:** in
`generate_logs.py`, scope normal file access so each user's baseline
department folder gets ~80-90% of their file events and other
departments' folders only occasionally, and bias email recipients
toward the same department most of the time. That would make graph
centrality (especially the foreign-folder edges) a genuinely
informative signal for `privilege_probe`, rather than just a feature
engineering job (Day 3) picking it up via burst counts. Flagging this
now rather than silently reporting a positive-looking number that
isn't actually true.

## Updated model comparison (Day 4 + 5)

| Model                      | Precision@6 | Recall@6 | F1@6 | ROC-AUC |
|----------------------------|:-----------:|:--------:|:----:|:-------:|
| Day 3 z-score baseline     | 1.0         | 1.0      | 1.0  | 1.0     |
| Isolation Forest (scratch) | 1.0         | 1.0      | 1.0  | 1.0     |
| Mahalanobis detector       | 1.0         | 1.0      | 1.0  | 1.0     |
| Autoencoder (scratch)      | 1.0         | 1.0      | 1.0  | 1.0     |

(Same Day 4 caveat applies: this is pipeline validation on a small,
clean-signal synthetic dataset, not a real-world accuracy claim.)

## Next (Day 6)

Explainable AI: SHAP and LIME to produce human-readable "why was this
user flagged" explanations. Since scikit-learn wasn't installable
here, SHAP/LIME (which also depend on it) will need either the same
from-scratch treatment or to be attempted fresh in case they have
lighter dependency footprints — need to check.
