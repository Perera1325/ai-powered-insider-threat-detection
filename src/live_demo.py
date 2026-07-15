"""
Live real-time demo mode.

Everything above (Days 1-7) works on a fixed, already-generated 30-day
dataset. This module exists purely for presentation purposes: it
simulates a small "company" of users generating fresh activity day by
day, in real time, so a lecturer/audience can watch a risk score climb
and an alert fire the moment an insider scenario is injected - rather
than only ever looking at a static, pre-computed result.

It deliberately reuses the exact same generator functions from Day 1
(`generate_logs.py`), the exact same injector functions from Day 1
(`inject_anomalies.py`), and the exact same feature-engineering
functions from Day 3 (`feature_engineering.py`) - so the "live" demo is
provably running the same pipeline as the rest of the project, not a
separate toy.

Scoring uses the model and standardization statistics (mean/std)
already fit on the original 100-user baseline (Day 4), exactly like a
real deployed system would: score new incoming behaviour against a
model trained on historical data, never retrain live on every event.

Usage: `run_live_demo()` is a generator - each `next()` advances one
simulated day ("tick") and yields a small dict describing the current
state (risk table, which user/scenario was just injected, etc). This
makes it trivial to both unit-test (iterate it in plain Python) and
wire into Streamlit (iterate it in a `for` loop, updating placeholders,
with a `time.sleep()` between ticks - see app.py's "Live Demo" tab).
"""

import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from generate_logs import generate_logon, generate_device, generate_http, generate_email, generate_file
from inject_anomalies import (
    inject_after_hours_login, inject_mass_file_access, inject_usb_exfiltration,
    inject_data_exfil_email, inject_privilege_probe, SCENARIOS,
)
from feature_engineering import (
    build_logon_features, build_device_features, build_http_features,
    build_email_features, build_file_features,
)

DEPARTMENTS = ["Finance", "Engineering", "HR", "Sales", "IT", "Legal", "Marketing"]


def create_live_cohort(n=12, seed=123):
    """A small, separate demo-only user cohort (ids prefixed 'L' so they
    never collide with the U0001..U0100 baseline population used to
    train the models)."""
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n):
        dept = DEPARTMENTS[rng.randint(0, len(DEPARTMENTS))]
        rows.append(
            {
                "user_id": f"L{i+1:02d}",
                "full_name": f"Demo User {i+1}",
                "department": dept,
                "role": "Employee",
                "pc": f"PC-DEMO-{i+1:03d}",
                "email": f"l{i+1:02d}@simcorp-internal.local",
            }
        )
    return pd.DataFrame(rows)


def _next_weekday(d):
    d = d + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _empty_event_store():
    return {
        "logon": pd.DataFrame(columns=["user_id", "pc", "activity", "timestamp"]),
        "device": pd.DataFrame(columns=["user_id", "pc", "activity", "timestamp"]),
        "http": pd.DataFrame(columns=["user_id", "pc", "url", "timestamp"]),
        "email": pd.DataFrame(columns=["user_id", "to", "external", "size_kb", "attachments", "timestamp"]),
        "file": pd.DataFrame(columns=["user_id", "pc", "filename", "action", "timestamp"]),
    }


def _simulate_one_day(users_df, day):
    return {
        "logon": generate_logon(users_df, num_days=1, start_date=day),
        "device": generate_device(users_df, num_days=1, start_date=day),
        "http": generate_http(users_df, num_days=1, start_date=day),
        "email": generate_email(users_df, num_days=1, start_date=day),
        "file": generate_file(users_df, num_days=1, start_date=day),
    }


def _apply_injection(store, users_df, demo_user_id, scenario, day):
    user_row = users_df[users_df.user_id == demo_user_id].iloc[0]

    if scenario == "after_hours_login":
        store["logon"] = inject_after_hours_login(store["logon"], demo_user_id, day)
    elif scenario == "mass_file_access":
        store["file"] = inject_mass_file_access(store["file"], demo_user_id, user_row.pc, day)
    elif scenario == "usb_exfiltration":
        store["device"], store["file"] = inject_usb_exfiltration(
            store["device"], store["file"], demo_user_id, user_row.pc, day
        )
    elif scenario == "data_exfil_email":
        store["email"] = inject_data_exfil_email(store["email"], demo_user_id, day)
    elif scenario == "privilege_probe":
        other_depts = [d for d in DEPARTMENTS if d != user_row.department]
        foreign_dept = other_depts[0]
        store["file"] = inject_privilege_probe(store["file"], demo_user_id, user_row.pc, day, foreign_dept)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return store


def compute_live_features(accumulated, users_df, feature_cols):
    """Runs the SAME Day 3 feature builders on whatever events have
    accumulated so far, and reindexes to the full feature_cols list
    (any feature with zero signal so far, e.g. no device events yet,
    naturally comes out as 0)."""
    logon_feats = build_logon_features(accumulated["logon"]) if not accumulated["logon"].empty else pd.DataFrame()
    device_feats = build_device_features(accumulated["device"]) if not accumulated["device"].empty else pd.DataFrame()
    http_feats = build_http_features(accumulated["http"]) if not accumulated["http"].empty else pd.DataFrame()
    email_feats = build_email_features(accumulated["email"]) if not accumulated["email"].empty else pd.DataFrame()
    file_feats = build_file_features(accumulated["file"]) if not accumulated["file"].empty else pd.DataFrame()

    features = users_df[["user_id"]].set_index("user_id")
    for feats in (logon_feats, device_feats, http_feats, email_feats, file_feats):
        if not feats.empty:
            features = features.join(feats)
    features = features.reindex(columns=feature_cols, fill_value=0).fillna(0).reset_index()
    return features


def score_live(features_df, feature_cols, train_mean, train_std, model):
    X_live = features_df[feature_cols].values
    Z = (X_live - train_mean) / np.where(train_std == 0, 1, train_std)
    scores = model.decision_function(Z)
    return scores


def run_live_demo(
    users_df, feature_cols, train_mean, train_std, model,
    n_ticks=10, inject_at_tick=6, demo_user_id="L01", scenario="mass_file_access",
    start_date=None, seed=42,
):
    """
    Generator - yields one dict per simulated day ("tick"):
      {tick, date, injected (bool, only True the tick it happens),
       scenario, demo_user_id, risk_table (DataFrame: user_id, score, risk_score)}
    """
    if start_date is None:
        start_date = datetime(2026, 9, 7)  # a Monday
    while start_date.weekday() >= 5:
        start_date += timedelta(days=1)

    accumulated = _empty_event_store()
    day = start_date

    for tick in range(1, n_ticks + 1):
        tick_events = _simulate_one_day(users_df, day)
        injected = tick == inject_at_tick
        if injected:
            tick_events = _apply_injection(tick_events, users_df, demo_user_id, scenario, day)

        for k in accumulated:
            if tick_events[k].empty:
                continue
            if accumulated[k].empty:
                accumulated[k] = tick_events[k]
            else:
                accumulated[k] = pd.concat([accumulated[k], tick_events[k]], ignore_index=True)

        features_df = compute_live_features(accumulated, users_df, feature_cols)
        scores = score_live(features_df, feature_cols, train_mean, train_std, model)
        lo, hi = scores.min(), scores.max()
        risk_score = 100 * (scores - lo) / (hi - lo + 1e-9)

        risk_table = pd.DataFrame(
            {"user_id": features_df.user_id, "anomaly_score": scores, "risk_score": risk_score.round(1)}
        ).sort_values("anomaly_score", ascending=False).reset_index(drop=True)

        yield {
            "tick": tick,
            "date": day,
            "injected": injected,
            "scenario": scenario if injected else None,
            "demo_user_id": demo_user_id,
            "risk_table": risk_table,
        }

        day = _next_weekday(day)


def plot_risk_history(history, demo_user_id, injected_at_tick):
    """
    history: list of dicts with keys 'tick' and 'demo_user_risk' (float)
    Draws the demo user's risk score over time, with a vertical marker
    at the tick the attack was injected.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ticks = [h["tick"] for h in history]
    risks = [h["demo_user_risk"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(ticks, risks, marker="o", color="#4a90d9", label=f"{demo_user_id} risk score")
    ax.axvline(injected_at_tick, color="#d94a4a", linestyle="--", label="attack injected here")
    ax.axhline(70, color="gray", linestyle=":", alpha=0.6, label="High-risk threshold")
    ax.set_xlabel("Simulated day (tick)")
    ax.set_ylabel("Risk score (0-100)")
    ax.set_ylim(-5, 105)
    ax.set_title(f"{demo_user_id}'s risk score over the live feed")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    return fig
