"""
Real-time event producer.

Plays the role of the organisation's log pipeline: it appends activity events
(logon / device / http / email / file) as JSON lines to  <dir>/events.jsonl,
one simulated working day per tick, every --interval seconds of REAL time.

File format (one JSON object per line, "type" says what it is):
  user   - one per user (org directory)
  event  - {"type":"event","kind":"logon"|"device"|"http"|"email"|"file", ...fields}
  truth  - an attack was injected (demo ground truth; a real system has no such line)
  tick   - end-of-batch marker {"tick": n, "sim_date": ..., "emit_ts": wall-clock}

A first "history" batch (tick 0, default 30 working days of normal behaviour) is
written immediately so the scorer's sliding window matches the feature window the
models were trained on. After that the stream is live.

Attacks are injected either automatically (--inject-after N) or on demand by
writing <dir>/control.json  {"nonce": "...", "user": "L03", "scenario": "mass_file_access"}
(the dashboard's "Inject attack now" button does exactly that).
Create <dir>/stop to end the stream early, <dir>/pause to pause it, and write seconds-per-day
into <dir>/interval to change the speed while it is running.

Run standalone:
  python src/realtime/event_producer.py --dir live --interval 2 --inject-after 8
"""

import argparse
import json
import os
import sys
import time
from datetime import timedelta

sys.path.insert(0, "src")
import pandas as pd

import generate_logs as gl
import live_demo as ld

KINDS = ["logon", "device", "http", "email", "file"]


def _write_df(fh, kind, df, emit_ts):
    if df is None or df.empty:
        return 0
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    out.insert(0, "kind", kind)
    out.insert(0, "type", "event")
    out["emit_ts"] = emit_ts
    fh.write(out.to_json(orient="records", lines=True).rstrip("\n") + "\n")
    return len(out)


def run_producer(out_dir="live", n_users=12, seed=123, interval=2.0, history_days=30, ticks=40,
                 inject_user=None, scenario="mass_file_access", inject_after=0, on_tick=None):
    """Runs the stream; returns the number of events written. Importable for tests."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "events.jsonl")
    control_path = os.path.join(out_dir, "control.json")
    stop_path = os.path.join(out_dir, "stop")
    pause_path = os.path.join(out_dir, "pause")
    speed_path = os.path.join(out_dir, "interval")
    for p in (path, control_path, stop_path, pause_path, speed_path):
        if os.path.exists(p):
            os.remove(p)

    gl.reseed(seed)
    users = ld.create_live_cohort(n=n_users, seed=seed)
    total = 0
    last_nonce = None

    with open(path, "a", encoding="utf-8") as fh:
        for rec in users.to_dict("records"):
            fh.write(json.dumps({"type": "user", **rec}) + "\n")

        # ---- tick 0: history backfill (normal behaviour only) ----
        hist_start = pd.Timestamp("2026-08-03").to_pydatetime()  # a Monday
        emit = time.time()
        gens = {
            "logon": gl.generate_logon, "device": gl.generate_device, "http": gl.generate_http,
            "email": gl.generate_email, "file": gl.generate_file,
        }
        last_ts = hist_start
        for kind, fn in gens.items():
            df = fn(users, num_days=history_days, start_date=hist_start)
            total += _write_df(fh, kind, df, emit)
            if not df.empty:
                last_ts = max(last_ts, pd.to_datetime(df.timestamp).max().to_pydatetime())
        day = ld._next_weekday(last_ts.replace(hour=0, minute=0, second=0, microsecond=0))
        fh.write(json.dumps({"type": "tick", "tick": 0, "sim_date": str(last_ts.date()), "emit_ts": emit}) + "\n")
        fh.flush()

        # ---- live ticks ----
        for tick in range(1, ticks + 1):
            while os.path.exists(pause_path) and not os.path.exists(stop_path):
                time.sleep(0.15)               # paused from the dashboard
            if os.path.exists(stop_path):
                break
            emit = time.time()
            events = ld._simulate_one_day(users, day)

            cmd = None
            if os.path.exists(control_path):
                try:
                    with open(control_path, encoding="utf-8") as cf:
                        c = json.load(cf)
                    if c.get("nonce") != last_nonce:
                        last_nonce, cmd = c.get("nonce"), c
                except (ValueError, OSError):
                    pass  # half-written control file; try again next tick
            if cmd is None and inject_user and inject_after and tick == inject_after:
                cmd = {"user": inject_user, "scenario": scenario}
            if cmd:
                events = ld._apply_injection(events, users, cmd["user"], cmd["scenario"], day)
                fh.write(json.dumps({"type": "truth", "user": cmd["user"], "scenario": cmd["scenario"],
                                     "sim_date": str(day.date()), "tick": tick, "emit_ts": emit}) + "\n")

            for kind in KINDS:
                total += _write_df(fh, kind, events[kind], emit)
            fh.write(json.dumps({"type": "tick", "tick": tick, "sim_date": str(day.date()), "emit_ts": emit}) + "\n")
            fh.flush()
            if on_tick:
                on_tick(tick)
            day = ld._next_weekday(day)
            try:  # live speed control: dashboard writes seconds-per-day into <dir>/interval
                with open(speed_path, encoding="utf-8") as sf:
                    interval = max(0.0, float(sf.read().strip()))
            except (OSError, ValueError):
                pass
            if interval:
                time.sleep(interval)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="live")
    ap.add_argument("--users", type=int, default=12)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--history-days", type=int, default=30)
    ap.add_argument("--ticks", type=int, default=40)
    ap.add_argument("--inject-user", default=None)
    ap.add_argument("--scenario", default="mass_file_access")
    ap.add_argument("--inject-after", type=int, default=0)
    a = ap.parse_args()
    n = run_producer(a.dir, a.users, a.seed, a.interval, a.history_days, a.ticks,
                     a.inject_user, a.scenario, a.inject_after)
    print(f"producer finished: {n} events written to {a.dir}/events.jsonl")


if __name__ == "__main__":
    main()
