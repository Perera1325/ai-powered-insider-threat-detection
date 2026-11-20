"""Real-time pipeline: producer -> JSONL log -> incremental sliding-window scorer -> alerts."""
import json
import os
import subprocess
import sys
import time

import pytest

import dashboard_utils as du
from realtime.event_producer import run_producer
from realtime.scorer import RealTimeScorer, risk_from_scores

EASY = ["after_hours_login", "mass_file_access", "usb_exfiltration", "data_exfil_email"]


@pytest.fixture(scope="module")
def rt_model(matrix):
    X = matrix[0]
    model = du.LazyModels(X)["Isolation Forest (sklearn)"]
    s = model.decision_function(X)
    return model, s.min(), s.max()


def _scorer(path, matrix, rt_model, **kw):
    X, cols, mean, std = matrix
    model, lo, hi = rt_model
    return RealTimeScorer(path, model, cols, mean, std, lo, hi, **kw)


def test_risk_scale_is_absolute_and_clipped():
    r = risk_from_scores([-1, 0, 0.5, 2], lo=0, hi=1)
    assert list(r.round(1)) == [0.0, 0.0, 50.0, 100.0]


def test_producer_log_format(tmp_path):
    d = str(tmp_path)
    n = run_producer(d, interval=0, ticks=3, history_days=10)
    lines = [json.loads(l) for l in open(os.path.join(d, "events.jsonl"))]
    types = {l["type"] for l in lines}
    assert {"user", "event", "tick"} <= types
    assert [l["tick"] for l in lines if l["type"] == "tick"] == [0, 1, 2, 3]
    assert n == sum(1 for l in lines if l["type"] == "event")
    assert {l["kind"] for l in lines if l["type"] == "event"} <= {"logon", "device", "http", "email", "file"}
    assert sum(1 for l in lines if l["type"] == "user") == 12


def test_scorer_reads_incrementally_and_ignores_half_written_line(tmp_path, matrix, rt_model):
    d = str(tmp_path)
    run_producer(d, interval=0, ticks=2)
    path = os.path.join(d, "events.jsonl")
    sc = _scorer(path, matrix, rt_model)
    first = sc.poll()
    assert first["tick"] == 2
    assert sc.poll() in (None, sc.last_state)  # nothing new
    off = sc.offset
    with open(path, "a") as f:
        f.write('{"type": "tick", "tick": 3, "sim_da')  # half-written, no newline
    assert sc.poll() in (None, sc.last_state) and sc.offset == off  # not consumed
    assert sc.history_df().index.tolist() == [0, 1, 2]


@pytest.mark.parametrize("scenario", EASY)
def test_attack_detected_within_two_ticks_with_no_false_alarms_before(tmp_path, matrix, rt_model, scenario):
    d = str(tmp_path)
    run_producer(d, interval=0, ticks=12, inject_user="L03", scenario=scenario, inject_after=8)
    sc = _scorer(os.path.join(d, "events.jsonl"), matrix, rt_model)
    sc.poll()
    hits = [a for a in sc.alerts if a["user_id"] == "L03"]
    assert hits and hits[0]["tick"] in (8, 9, 10), sc.alerts
    assert not [a for a in sc.alerts if a["tick"] < 8], "false alarm before the attack"
    assert sc.truth and sc.truth[0]["user"] == "L03"


def test_quiet_stream_raises_no_alert(tmp_path, matrix, rt_model):
    d = str(tmp_path)
    run_producer(d, interval=0, ticks=12)  # nobody attacks
    sc = _scorer(os.path.join(d, "events.jsonl"), matrix, rt_model)
    sc.poll()
    assert sc.alerts == []


def test_control_file_injects_on_demand(tmp_path):
    d = str(tmp_path)
    def drop_control(tick):
        if tick == 2:
            with open(os.path.join(d, "control.json"), "w") as f:
                json.dump({"nonce": "n1", "user": "L05", "scenario": "usb_exfiltration"}, f)

    run_producer(d, interval=0, ticks=5, on_tick=drop_control)
    truth = [json.loads(l) for l in open(os.path.join(d, "events.jsonl")) if '"truth"' in l]
    assert len(truth) == 1 and truth[0]["user"] == "L05" and truth[0]["tick"] == 3


def test_end_to_end_subprocess_stream_alert_latency(tmp_path, matrix, rt_model):
    """Real separate producer process, wall-clock pacing; alert must arrive within seconds of the logs."""
    d = str(tmp_path)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.Popen(
        [sys.executable, os.path.join(root, "src/realtime/event_producer.py"), "--dir", d, "--interval", "0.4",
         "--ticks", "10", "--inject-user", "L03", "--scenario", "mass_file_access", "--inject-after", "5"],
        cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        sc = _scorer(os.path.join(d, "events.jsonl"), matrix, rt_model)
        deadline = time.time() + 90
        while time.time() < deadline and not any(a["user_id"] == "L03" for a in sc.alerts):
            sc.poll()
            time.sleep(0.1)
        hits = [a for a in sc.alerts if a["user_id"] == "L03"]
        assert hits, "no alert from the live stream"
        assert hits[0]["latency_s"] < 5.0, hits[0]  # seconds from log write to alert
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_scorer_exposes_feed_latency_and_alert_drivers(tmp_path, matrix, rt_model):
    run_producer(str(tmp_path), interval=0, ticks=6, inject_user="L03", scenario="mass_file_access", inject_after=3)
    sc = _scorer(str(tmp_path / "events.jsonl"), matrix, rt_model)
    sc.poll()
    feed = sc.feed_df(10)
    assert len(feed) == 10 and list(feed.columns) == ["time", "kind", "user_id", "detail"]
    lat = sc.latency_stats()
    assert lat["p50"] is not None and lat["p95"] >= lat["p50"]
    assert sc.live_events > 0 and sc.throughput() > 0
    assert all(s["score_ms"] >= 0 for s in sc.tick_stats)
    assert sc.alerts and "sd)" in sc.alerts[0]["drivers"]
    assert sc.last_state["active_alerts"] >= 1


def test_producer_finishes_without_pause_file(tmp_path):
    import threading
    d = str(tmp_path)
    os.makedirs(d, exist_ok=True)
    t = threading.Thread(target=run_producer, kwargs=dict(out_dir=d, interval=0, ticks=3))
    t.start(); t.join(timeout=60)
    assert not t.is_alive()  # no pause file -> finishes
