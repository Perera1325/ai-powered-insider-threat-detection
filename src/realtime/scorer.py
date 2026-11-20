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
from collections import deque

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
        self.feed = deque(maxlen=300)              # most recent raw events (for the live feed)
        self.tick_stats = []                       # per tick: events, latency_s, score_ms
        self._tick_events = 0
        self._live_t0 = None                       # wall-clock when tick 0 (history) was scored
        self.live_events = 0

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
                self._tick_events += 1
                self.feed.append({"time": rec.get("timestamp"), "kind": rec["kind"], "user_id": rec.get("user_id"),
                                  "detail": self._detail(rec)})
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

    @staticmethod
    def _detail(rec):
        k = rec["kind"]
        if k in ("logon", "device"):
            return str(rec.get("activity", ""))
        if k == "http":
            return str(rec.get("url", ""))
        if k == "email":
            return f"{'EXTERNAL ' if rec.get('external') else ''}{rec.get('size_kb', 0)} KB, {rec.get('attachments', 0)} attach."
        return f"{rec.get('action', '')} {str(rec.get('filename', '')).split(chr(92))[-1]}"

    def feed_df(self, n=15):
        return pd.DataFrame(list(self.feed)[-n:][::-1], columns=["time", "kind", "user_id", "detail"])

    def latency_stats(self):
        lat = [t["latency_s"] for t in self.tick_stats if t["tick"] > 0]
        if not lat:
            return {"p50": None, "p95": None, "max": None}
        return {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)), "max": float(max(lat))}

    def throughput(self):
        """Live events per wall-clock second since the history batch was scored."""
        if self._live_t0 is None:
            return 0.0
        return self.live_events / max(time.time() - self._live_t0, 1e-6)

    # ---- scoring ----
    def _score(self, tick_rec):
        t_start = time.perf_counter()
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
        zmap = {u: Z[i] for i, u in enumerate(feats.user_id)}
        new_alerts = []
        for uid, r in zip(table.user_id, table.risk_score):
            if r >= self.threshold and uid not in self._active:
                self._active.add(uid)
                a = {"user_id": uid, "tick": tick_rec["tick"], "sim_date": tick_rec["sim_date"],
                     "risk_score": float(r), "latency_s": round(now - tick_rec["emit_ts"], 3)}
                z = pd.Series(zmap[uid], index=self.feature_cols)
                top = z.reindex(z.abs().sort_values(ascending=False).index).head(3)
                a["drivers"] = ", ".join(f"{k} ({v:+.1f} sd)" for k, v in top.items())
                new_alerts.append(a)
                self.alerts.append(a)
            elif r < self.rearm_below and uid in self._active:
                self._active.discard(uid)

        if tick_rec["tick"] == 0:
            self._live_t0 = now
        else:
            self.live_events += self._tick_events
        self.tick_stats.append({"tick": tick_rec["tick"], "events": self._tick_events,
                                "latency_s": round(now - tick_rec["emit_ts"], 3),
                                "score_ms": round(1000 * (time.perf_counter() - t_start), 1)})
        self._tick_events = 0
        self.history.append({"tick": tick_rec["tick"], "sim_date": tick_rec["sim_date"], **dict(zip(table.user_id, table.risk_score))})
        self.last_state = {
            "tick": tick_rec["tick"], "sim_date": tick_rec["sim_date"], "risk_table": table,
            "new_alerts": new_alerts, "n_events": self.n_events, "window_events": sum(len(v) for v in self.window.values()),
            "active_alerts": len(self._active),
        }
        return self.last_state

    def history_df(self):
        return pd.DataFrame(self.history).set_index("tick") if self.history else pd.DataFrame()
