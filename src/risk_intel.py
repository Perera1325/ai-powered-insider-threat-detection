"""Risk intelligence: multi-model consensus, department roll-up and population drift (PSI).

Pure functions, no Streamlit. Nothing here uses the ground-truth labels."""
import numpy as np
import pandas as pd

ALERT_THRESHOLD = 70.0


def consensus_risk(risks, user_ids, threshold=ALERT_THRESHOLD):
    """risks: {model name: 0-100 risk array aligned with user_ids}.
    Returns one row per user: consensus (mean risk), spread (std across models), models_alerting and a
    confidence label (High = most models agree the user is above the threshold)."""
    names = list(risks)
    M = np.vstack([np.asarray(risks[m], float) for m in names])  # models x users
    n = len(names)
    alerting = (M >= threshold).sum(axis=0)
    share = alerting / n
    conf = np.where(share >= 0.75, "High", np.where(share >= 0.4, "Medium", np.where(alerting > 0, "Low", "-")))
    df = pd.DataFrame({"user_id": list(user_ids), "consensus": M.mean(axis=0).round(1), "spread": M.std(axis=0).round(1),
                       "models_alerting": alerting.astype(int), "models_total": n, "confidence": conf})
    for i, m in enumerate(names):
        df[m] = M[i].round(1)
    return df.sort_values(["consensus", "spread"], ascending=[False, True]).reset_index(drop=True)


def department_risk(users, user_ids, risk, threshold=ALERT_THRESHOLD):
    """Risk roll-up per department: headcount, mean/max risk, users above the threshold."""
    r = pd.DataFrame({"user_id": list(user_ids), "risk": np.asarray(risk, float)})
    m = r.merge(users[["user_id", "department"]], on="user_id", how="left")
    m["department"] = m["department"].fillna("Unknown")
    g = m.groupby("department")
    out = pd.DataFrame({"users": g.size(), "mean_risk": g.risk.mean().round(1), "max_risk": g.risk.max().round(1),
                        "high_risk_users": g.risk.apply(lambda s: int((s >= threshold).sum()))})
    out["high_risk_share"] = (out.high_risk_users / out.users).round(3)
    return out.sort_values(["high_risk_users", "mean_risk"], ascending=False).reset_index().rename(columns={"index": "department"})


def psi(expected, actual, bins=10):
    """Population Stability Index of one feature (baseline vs new). Bins come from the baseline quantiles.
    < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 significant shift."""
    e, a = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # constant (or nearly constant) baseline feature
        return 0.0 if np.allclose(a, e[0]) else 1.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def drift_status(value):
    return "Stable" if value < 0.10 else ("Moderate" if value < 0.25 else "Significant")


def feature_drift(baseline, new, feature_cols, bins=10):
    """PSI per feature, largest shift first. baseline/new are feature frames sharing feature_cols."""
    rows = []
    for c in feature_cols:
        if c in baseline.columns and c in new.columns and len(new) and len(baseline):
            v = psi(baseline[c].values, new[c].values, bins)
            rows.append({"feature": c, "psi": round(v, 3), "status": drift_status(v),
                         "baseline_mean": round(float(baseline[c].mean()), 2), "new_mean": round(float(new[c].mean()), 2)})
    return pd.DataFrame(rows, columns=["feature", "psi", "status", "baseline_mean", "new_mean"]) \
        .sort_values("psi", ascending=False).reset_index(drop=True)
