"""
Real-time (micro-batch) scorer.

Tails <dir>/events.jsonl incrementally (byte offset, so each poll only reads NEW
lines), keeps a sliding window of the last `window_days` simulated days per
event type, rebuilds the SAME per-user features used in training
(feature_engineering.py) over that window, scores them with an already-trained
model, and raises an alert when a user's risk crosses the threshold.

Risk is on an ABSOLUTE scale: (score - lo) / (hi - lo) * 100 where lo/hi are the
min/max anomaly scores of the training population - so quiet users stay low and
an attack is not hidden by (or relative to) the other users in the stream.
(The older Live Demo tab normalises within the 12-user cohort, so someone is
always at 100.)

Latency = wall-clock seconds from the producer writing a batch (emit_ts) to the
scorer raising the alert for it.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import live_demo as ld

KINDS = ["logon", "device", "http", "email", "file"]


def risk_from_scores(scores, lo, hi):
    return np.clip(100 * (np.asarray(scores) - lo) / (hi - lo + 1e-9), 0, 100)


class RealTimeScorer:
    def __init__(self, events_path, model, feature_cols, train_mean, train_std, score_lo, score_hi,
                 window_days=30, alert_threshold=70.0, rearm_below=50.0):
        self.path = events_path
        self.model, self.feature_cols = model, feature_cols
        self.mean, self.std = np.asarray(train_mean), np.asarray(train_std)
        self.lo, self.hi = float(score_lo), float(score_hi)
        self.window_days, self.threshold, self.rearm_below = window_days, alert_threshold, rearm_below
        self.offset = 0
        self.users = None
        self.store = {k: [] for k in KINDS}       # list of DataFrame chunks
        self.window = ld._empty_event_store()
        self.n_events = 0
        self.history = []                          # rows: tick, sim_date, {user: risk}
        self.alerts, self.truth = [], []
        self._active = set()
        self.last_state = None

    # ---- reading ----
    def _read_new_lines(self):
        if not os.path.exists(self.path):
            return []
        with open(self.path, "rb") as fh:
            fh.seek(self.offset)
            data = fh.read()
        cut = data.rfind(b"\n")
        if cut < 0:
            return []
        self.offset += cut + 1
        return [json.loads(l) for l in data[: cut + 1].decode("utf-8").splitlines() if l.strip()]

    def _flush(self, chunks):
        for kind, rows in chunks.items():
            if rows:
                df = pd.DataFrame(rows).drop(columns=["type", "kind", "emit_ts"], errors="ignore")
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                self.window[kind] = df if self.window[kind].empty else pd.concat([self.window[kind], df], ignore_index=True)
            chunks[kind] = []

    def poll(self):
        """Consume new complete lines, scoring after EVERY tick marker in order (so history
        is complete even if the caller polls slowly). Returns the latest state dict (with
        all alerts raised during this poll in state['new_alerts']) or None if nothing new."""
        lines = self._read_new_lines()
        if not lines:
            return None
        chunks = {k: [] for k in KINDS}
        new_alerts, state = [], None
        for rec in lines:
            t = rec.get("type")
            if t == "user":
                rec = {k: v for k, v in rec.items() if k != "type"}
                self.users = pd.concat([self.users, pd.DataFrame([rec])], ignore_index=True) if self.users is not None else pd.DataFrame([rec])
            elif t == "event":
                chunks[rec["kind"]].append(rec)
                self.n_events += 1
            elif t == "truth":
                self.truth.append(rec)
            elif t == "tick":
                self._flush(chunks)
                state = self._score(rec)
                new_alerts += state["new_alerts"]
        if state is None:
            return self.last_state
        state["new_alerts"] = new_alerts
        return state

    # ---- scoring ----
    def _score(self, tick_rec):
        sim_end = pd.Timestamp(tick_rec["sim_date"])
        cutoff = sim_end - pd.Timedelta(days=self.window_days)
        for kind in KINDS:
            df = self.window[kind]
            if not df.empty:
                self.window[kind] = df[df["timestamp"] >= cutoff].reset_index(drop=True)

        feats = ld.compute_live_features(self.window, self.users, self.feature_cols)
        Z = (feats[self.feature_cols].values - self.mean) / np.where(self.std == 0, 1, self.std)
        scores = self.model.decision_function(Z)
        risk = risk_from_scores(scores, self.lo, self.hi)
        table = pd.DataFrame({"user_id": feats.user_id, "anomaly_score": scores, "risk_score": risk.round(1)}) \
            .sort_values("risk_score", ascending=False).reset_index(drop=True)

        now = time.time()
        new_alerts = []
        for uid, r in zip(table.user_id, table.risk_score):
            if r >= self.threshold and uid not in self._active:
                self._active.add(uid)
                a = {"user_id": uid, "tick": tick_rec["tick"], "sim_date": tick_rec["sim_date"],
                     "risk_score": float(r), "latency_s": round(now - tick_rec["emit_ts"], 3)}
                new_alerts.append(a)
                self.alerts.append(a)
            elif r < self.rearm_below and uid in self._active:
                self._active.discard(uid)

        self.history.append({"tick": tick_rec["tick"], "sim_date": tick_rec["sim_date"], **dict(zip(table.user_id, table.risk_score))})
        self.last_state = {
            "tick": tick_rec["tick"], "sim_date": tick_rec["sim_date"], "risk_table": table,
            "new_alerts": new_alerts, "n_events": self.n_events, "window_events": sum(len(v) for v in self.window.values()),
        }
        return self.last_state

    def history_df(self):
        return pd.DataFrame(self.history).set_index("tick") if self.history else pd.DataFrame()
