"""
Day 3 - Quick z-score anomaly ranking sanity check.

Not a real model (that's Day 4 - Isolation Forest / One-Class SVM), just
a fast preview: standardize every engineered feature, sum the absolute
z-scores per user, and see whether the 6 known malicious users rise
toward the top of that ranking. This tells us whether the Day 3 feature
set carries useful separating signal before we invest in real models.
"""

import numpy as np
import pandas as pd

DATA_DIR = "data/processed/full"


def main():
    df = pd.read_csv(f"{DATA_DIR}/user_features_preview_with_labels.csv")
    feature_cols = [
        c for c in df.columns if c not in ("user_id", "is_malicious", "scenario", "injected_day")
    ]

    z = (df[feature_cols] - df[feature_cols].mean()) / df[feature_cols].std(ddof=0).replace(0, 1)
    df["anomaly_score"] = z.abs().sum(axis=1)

    ranked = df.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1

    malicious_ranks = ranked[ranked.is_malicious == True][  # noqa: E712
        ["rank", "user_id", "scenario", "anomaly_score"]
    ]

    n_users = len(df)
    n_malicious = int(df.is_malicious.sum())

    lines = ["# Day 3 Feature Ranking Sanity Check\n"]
    lines.append(
        f"Simple z-score-sum ranking over all {len(feature_cols)} engineered "
        f"features (not a real model - see Day 4). Out of {n_users} users, "
        f"{n_malicious} are known malicious insiders.\n"
    )
    lines.append("## Where malicious users land in the ranking\n")
    lines.append("```")
    lines.append(malicious_ranks.to_string(index=False))
    lines.append("```\n")

    top10_hits = ranked.head(10).is_malicious.sum()
    lines.append(
        f"{top10_hits}/{n_malicious} malicious users appear in the top 10 of "
        f"{n_users} by this naive combined z-score - "
        f"a real model (Day 4) with proper feature weighting/learning should "
        "do at least this well, likely better since it can learn non-linear "
        "combinations rather than a flat sum of z-scores.\n"
    )

    lines.append("## Top 10 ranked users overall\n")
    lines.append("```")
    lines.append(
        ranked.head(10)[["rank", "user_id", "anomaly_score", "is_malicious", "scenario"]].to_string(
            index=False
        )
    )
    lines.append("```")

    with open("docs/day3_ranking_check.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
