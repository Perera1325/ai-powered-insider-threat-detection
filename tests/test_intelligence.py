"""Response playbooks + SLAs, risk intelligence (consensus, department, drift) and the executive overview."""
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import executive
import playbook
import risk_intel
import soc_tools as soc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T0 = datetime(2026, 11, 27, 9, 0, 0)


def _case(entries):
    """entries: [(hours after T0, status)] -> case dict."""
    c = {"status": entries[-1][1], "history": []}
    for h, s in entries:
        c["history"].append({"time": (T0 + timedelta(hours=h)).strftime("%Y-%m-%d %H:%M:%S"), "status": s, "note": "", "analyst": "a"})
    return c


# ------------------------------------------------------------------ playbook
def test_playbook_high_includes_technique_steps_in_phase_order():
    p = playbook.build_playbook("High", ["T1052.001", "T1048"])
    assert list(p.step) == list(range(1, len(p) + 1))
    assert [playbook.PHASES.index(x) for x in p.phase] == sorted(playbook.PHASES.index(x) for x in p.phase)
    assert p.action.is_unique
    text = " ".join(p.action)
    assert "removable media" in text and "external e-mails" in text


def test_playbook_low_has_no_technique_steps_and_unknown_severity_falls_back():
    assert len(playbook.build_playbook("Low", ["T1048"])) == len(playbook.SEVERITY_STEPS["Low"])
    assert len(playbook.build_playbook("???", [])) == len(playbook.SEVERITY_STEPS["Low"])
    assert (playbook.build_playbook("High").why.str.startswith("Severity")).all()


def test_sla_hours_by_severity():
    assert [playbook.sla_hours(s) for s in ("High", "Medium", "Low")] == [4, 24, 72]


@pytest.mark.parametrize("entries,now_h,sev,state", [
    ([], 1, "High", "Untriaged"),
    ([(0, "Investigating")], 1, "High", "On track"),
    ([(0, "Investigating")], 3.5, "High", "At risk"),       # 0.5 h left of 4 h (< 25%)
    ([(0, "Escalated")], 5, "High", "Breached"),
    ([(0, "Investigating"), (3, "Closed - confirmed")], 99, "High", "Met"),
    ([(0, "Investigating"), (6, "Closed - false positive")], 99, "High", "Breached"),
    ([(0, "Investigating")], 10, "Medium", "On track"),
])
def test_case_sla_states(entries, now_h, sev, state):
    case = _case(entries) if entries else None
    assert playbook.case_sla(case, sev, T0 + timedelta(hours=now_h))["state"] == state


def test_sla_table_and_kpis():
    alerts = pd.DataFrame({"user_id": ["U1", "U2", "U3", "U4"], "risk_level": ["High", "High", "Medium", "High"],
                           "risk_score": [95, 88, 55, 80]})
    store = {"U1": _case([(0, "Investigating")]), "U2": _case([(0, "Investigating"), (2, "Closed - confirmed")]),
             "U3": _case([(0, "Escalated")])}
    now = T0 + timedelta(hours=5)
    sla = playbook.sla_table(alerts, store, now)
    assert dict(zip(sla.user_id, sla.state)) == {"U1": "Breached", "U2": "Met", "U3": "On track", "U4": "Untriaged"}
    k = executive.executive_kpis(alerts, store, sla)
    assert (k["alerts"], k["high"], k["untriaged_high"]) == (4, 3, 1)
    assert (k["open_cases"], k["escalated"], k["closed"], k["sla_breached"]) == (2, 1, 1, 1)
    assert k["mttr_hours"] == 2.0


def test_incident_brief_contains_evidence_and_checklist(data):
    feats = data["features"]
    cols = [c for c in feats.columns if c != "user_id"]
    uid = feats.user_id.iloc[0]
    dev = soc.feature_deviations(feats, uid, cols, 3)
    import mitre
    m = mitre.for_drivers(soc.feature_deviations(feats, uid, cols, 10))
    plan = playbook.build_playbook("High", m.technique.tolist())
    md = playbook.incident_brief(uid, 91.2, "High", "New", m, dev, plan, generated=T0)
    assert md.startswith(f"# Incident brief: {uid}") and "91.2/100" in md and "2026-11-27 09:00" in md
    assert "## Response checklist" in md and "1. **Triage**" in md
    assert all(f in md for f in dev.feature)


# ------------------------------------------------------------------ executive
def test_case_funnel_lists_every_status_and_mttr_none_when_empty():
    f = executive.case_funnel({"U1": _case([(0, "Escalated")])})
    assert list(f.status) == soc.STATUSES and f.set_index("status").cases["Escalated"] == 1 and f.cases.sum() == 1
    assert executive.mean_time_to_resolve({}) is None


def test_executive_summary_text():
    k = {"alerts": 15, "high": 6, "untriaged_high": 2, "open_cases": 3, "escalated": 1, "closed": 4,
         "sla_breached": 1, "sla_at_risk": 0, "mttr_hours": None}
    top = pd.DataFrame({"user_id": ["U1"], "risk_score": [97.3], "risk_level": ["High"], "case_status": ["New"]})
    dept = pd.DataFrame({"department": ["IT"], "users": [10], "mean_risk": [41.2], "high_risk_users": [3]})
    md = executive.executive_summary(k, top, dept, "Mahalanobis detector", generated=T0)
    assert "6 high severity (2 not yet triaged)" in md and "mean time to resolve n/a" in md
    assert "U1: 97.3 (High)" in md and "IT: 3 of 10" in md


# ------------------------------------------------------------------ risk intelligence
def test_consensus_confidence_levels():
    risks = {"a": [90, 90, 10, 80], "b": [95, 20, 10, 10], "c": [85, 15, 5, 10], "d": [99, 10, 0, 20]}
    c = risk_intel.consensus_risk(risks, ["U1", "U2", "U3", "U4"]).set_index("user_id")
    assert c.loc["U1", "confidence"] == "High" and c.loc["U1", "models_alerting"] == 4
    assert c.loc["U2", "confidence"] == "Low" and c.loc["U4", "confidence"] == "Low"
    assert c.loc["U3", "confidence"] == "-"
    assert risk_intel.consensus_risk(risks, ["U1", "U2", "U3", "U4"]).user_id.iloc[0] == "U1"


def test_consensus_on_real_models(data, matrix, models):
    import dashboard_utils as du
    X = matrix[0]
    risks = {m: du.score_users(models, m, X)[1] for m in du.MODEL_CHOICES}
    c = risk_intel.consensus_risk(risks, data["features"].user_id)
    assert len(c) == len(data["features"]) and c.consensus.between(0, 100).all()
    top = set(c.head(10).user_id)
    insiders = set(data["labels"][data["labels"].is_malicious.astype(bool)].user_id)
    assert len(top & insiders) >= 5  # the consensus finds most injected insiders


def test_department_risk_rollup():
    users = pd.DataFrame({"user_id": ["U1", "U2", "U3"], "department": ["IT", "IT", "HR"]})
    d = risk_intel.department_risk(users, ["U1", "U2", "U3"], [90, 30, 10]).set_index("department")
    assert d.loc["IT", "users"] == 2 and d.loc["IT", "high_risk_users"] == 1 and d.loc["IT", "mean_risk"] == 60.0
    assert d.loc["HR", "high_risk_share"] == 0.0


def test_psi_identical_shifted_and_constant():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 2000)
    assert risk_intel.psi(a, rng.normal(0, 1, 2000)) < 0.1
    assert risk_intel.psi(a, rng.normal(2, 1, 2000)) > 0.25
    assert risk_intel.psi(np.ones(50), np.ones(50)) == 0.0 and risk_intel.psi(np.ones(50), np.full(50, 3.0)) == 1.0
    assert [risk_intel.drift_status(v) for v in (0.05, 0.2, 0.4)] == ["Stable", "Moderate", "Significant"]


def test_feature_drift_orders_by_psi_and_flags_the_shifted_feature():
    rng = np.random.default_rng(1)
    base = pd.DataFrame({"x": rng.normal(0, 1, 500), "y": rng.normal(0, 1, 500)})
    new = pd.DataFrame({"x": rng.normal(0, 1, 500), "y": rng.normal(3, 1, 500)})
    d = risk_intel.feature_drift(base, new, ["x", "y", "missing"])
    assert list(d.feature) == ["y", "x"] and d.status.iloc[0] == "Significant"


# ------------------------------------------------------------------ the real app
@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=600)
    at.run()
    return at


def test_new_tabs_render_without_errors(app):
    assert not app.exception, [e.value for e in app.exception]
    heads = [h.value for h in app.subheader]
    for want in ("Executive overview", "Response playbooks and SLAs", "Risk intelligence"):
        assert want in heads
    labels = [m.label for m in app.metric]
    for want in ("High-severity alerts", "Mean time to resolve", "SLA", "High-confidence alerts"):
        assert want in labels


def test_playbook_start_case_button_starts_sla_clock(app):
    btn = [b for b in app.button if b.label == "Acknowledge and start case"][0]
    assert not btn.disabled
    btn.click().run()
    assert not app.exception, [e.value for e in app.exception]
    state = [m for m in app.metric if m.label == "SLA"][0]
    assert state.value in ("On track", "At risk")  # clock started


def test_drift_department_view(app):
    assert any(m.label == "Significant shifts" for m in app.metric)


def test_small_sample_drift_is_within_noise_and_real_shift_survives():
    rng = np.random.default_rng(2)
    base = pd.DataFrame({"x": rng.normal(0, 1, 400), "y": rng.normal(0, 1, 400)})
    new = pd.DataFrame({"x": rng.normal(0, 1, 15), "y": rng.normal(4, 1, 15)})
    d = risk_intel.feature_drift(base, new, ["x", "y"]).set_index("feature")
    assert d.loc["x", "status"] == "Within noise"
    assert d.loc["y", "status"] == "Significant"
