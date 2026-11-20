"""Unit tests for analyst-workflow helpers + the app features built on them."""
import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import dashboard_utils as du
import soc_tools as soc

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


@pytest.fixture(scope="module")
def ctx():
    d = du.load_data()
    X, fc, _, _ = du.get_feature_matrix(d["features"])
    return d, X, fc, du.LazyModels(X)


def test_case_lifecycle_and_roundtrip():
    s = soc.new_case_store()
    assert soc.case_status(s, "U1") == "New"
    soc.update_case(s, "U1", "Investigating", "checked VPN logs", now=datetime(2026, 1, 1, 9))
    soc.update_case(s, "U1", "Closed - false positive", "", now=datetime(2026, 1, 2, 9))
    assert soc.case_status(s, "U1") == "Closed - false positive"
    assert soc.latest_note(s, "U1") == "checked VPN logs"
    with pytest.raises(ValueError):
        soc.update_case(s, "U1", "bogus")
    back = soc.cases_from_frame(soc.cases_to_frame(s))
    assert back["U1"]["status"] == "Closed - false positive" and len(back["U1"]["history"]) == 2
    assert soc.cases_from_frame(pd.DataFrame({"x": [1]})) == {}


def test_threshold_sweep_monotonic_and_correct():
    risk = np.array([95, 90, 60, 50, 20, 10])
    y = np.array([1, 1, 0, 1, 0, 0])
    sw = soc.threshold_sweep(risk, y, [0, 55, 92, 100])
    assert list(sw.alerts) == [6, 3, 1, 0]
    assert sw.recall.iloc[0] == 1.0 and sw.recall.iloc[-1] == 0.0
    assert sw.loc[sw.threshold == 55, "precision"].iloc[0] == pytest.approx(2 / 3)
    assert sw.loc[sw.threshold == 55, "missed_insiders"].iloc[0] == 1
    full = soc.threshold_sweep(risk, y)
    assert (np.diff(full.alerts.values) <= 0).all()


def test_drilldown_helpers(ctx):
    d, X, fc, _ = ctx
    bad = d["labels"][d["labels"].is_malicious].user_id.iloc[0]
    dev = soc.feature_deviations(d["features"], bad, fc, 5)
    assert len(dev) == 5 and dev.z_score.abs().is_monotonic_decreasing
    tl = soc.activity_timeline(d["file_df"], d["email_df"], bad)
    assert set(tl.series) == {"File events", "E-mails sent", "External e-mails"} and (tl["count"] >= 0).all()
    peer = soc.peer_comparison(d["features"], d["users"], bad, fc, 4)
    assert len(peer) == 4 and peer.department_mean.notna().all()


def test_model_agreement(ctx):
    d, X, fc, m = ctx
    names = ["Isolation Forest (sklearn)", "Mahalanobis detector"]
    ranks, ov, cons = soc.model_rankings(lambda n: du.score_users(m, n, X)[1], names, list(d["features"].user_id), 10)
    assert ov.shape == (2, 2) and (np.diag(ov.values) == 1).all()
    assert ranks.min().min() == 1 and ranks.max().max() == len(d["features"])
    assert cons.models_flagging.max() <= 2


def test_exports(ctx):
    d, X, fc, m = ctx
    alerts = du.build_alert_table(d["features"], d["labels"], m, "Isolation Forest (sklearn)", X, top_k=5)
    s = soc.new_case_store()
    soc.update_case(s, alerts.user_id.iloc[0], "Escalated", "needs HR <review> & legal")
    csv = soc.alerts_csv(alerts, s).decode()
    assert "case_status" in csv and "Escalated" in csv
    pdf = soc.alerts_pdf(alerts, s, "Isolation Forest (sklearn)", {alerts.user_id.iloc[0]: "x (+3 sd)"})
    assert pdf[:5] == b"%PDF-" and len(pdf) > 2000


def test_app_case_save_and_investigate_and_simulator():
    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    uid = at.selectbox(key="inv_uid").value
    # save a case on the investigate tab
    at.selectbox(key=f"inv_status_{uid}").select("Investigating")
    at.text_input(key=f"inv_note_{uid}").input("reviewing")
    [b for b in at.button if b.key == f"inv_save_{uid}"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["cases"][uid]["status"] == "Investigating"
    assert [m.value for m in at.metric if m.label == "Open cases"] == ["1"], [(m.label, m.value) for m in at.metric][:8]
    # threshold simulator responds to the slider
    at.slider(key="sim_thr").set_value(20).run()
    lo = int([m for m in at.metric if m.label == "Users alerted"][0].value)
    at.slider(key="sim_thr").set_value(90).run()
    hi = int([m for m in at.metric if m.label == "Users alerted"][0].value)
    assert not at.exception and lo > hi
