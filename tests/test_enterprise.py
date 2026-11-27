"""Login + roles, SQLite persistence, notifications and the MITRE ATT&CK mapping."""
import os

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import auth
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import case_store as cs
import mitre
import notify
import soc_tools as soc


# ------------------------------------------------------------------ auth
def test_password_hash_roundtrip_and_salt():
    h1, h2 = auth.hash_password("s3cret"), auth.hash_password("s3cret")
    assert h1 != h2 and auth.verify_password("s3cret", h1) and not auth.verify_password("nope", h1)
    assert not auth.verify_password("x", "garbage")


def test_authenticate_and_permissions():
    users = auth.demo_users()
    assert auth.authenticate("analyst", "analyst123", users) == "analyst"
    assert auth.authenticate("analyst", "wrong", users) is None and auth.authenticate("ghost", "x", users) is None
    assert auth.can("admin", "admin") and not auth.can("analyst", "admin")
    assert auth.can("analyst", "case_write") and not auth.can("viewer", "case_write")
    assert auth.can("viewer", "view") and not auth.can("viewer", "export")


def test_normalise_users_skips_bad_roles_and_hashes_plaintext():
    u = auth.normalise_users({"a": {"password": "p", "role": "admin"}, "b": {"password": "p", "role": "root"}, "c": {"role": "viewer"}})
    assert list(u) == ["a"] and u["a"]["password_hash"].startswith("pbkdf2_sha256$")


# ------------------------------------------------------------------ database
def test_case_history_persists_and_restores(tmp_path):
    db = str(tmp_path / "t.db")
    cs.add_case_entry("U0001", "Investigating", "checking", "analyst", path=db)
    cs.add_case_entry("U0001", "Escalated", "confirmed odd", "admin", path=db)
    store = cs.load_store(db)
    assert store["U0001"]["status"] == "Escalated" and len(store["U0001"]["history"]) == 2
    cs.replace_cases({"U9": {"status": "New", "history": []}}, db)
    assert cs.load_store(db) == {}


def test_audit_and_notification_dedupe(tmp_path):
    db = str(tmp_path / "t.db")
    cs.log_audit("admin", "admin", "login", "ok", path=db)
    assert cs.audit_frame(path=db).action.tolist() == ["login"]
    a = {"tick": 4, "user_id": "L03", "risk_score": 91.0, "latency_s": 0.4, "drivers": "copy_count (+9.0 sd)", "stream": "s1"}
    assert notify.notify_alert(a, "T1005", path=db) is True
    assert notify.notify_alert(a, "T1005", path=db) is False  # same alert is not raised twice
    assert len(cs.notifications_frame(path=db)) == 1


def test_escalation_notifies_other_statuses_do_not(tmp_path):
    db = str(tmp_path / "t.db")
    assert notify.notify_case("U1", "Investigating", "a", path=db) is False
    assert notify.notify_case("U1", "Escalated", "a", "bad", path=db) is True


def test_webhook_url_must_be_https_and_comes_from_config(monkeypatch):
    monkeypatch.delenv("ITD_WEBHOOK_URL", raising=False)
    assert notify.webhook_url() == "" and notify.webhook_url("http://x.example/hook") == ""
    assert notify.webhook_url("https://hooks.example/abc") == "https://hooks.example/abc"
    assert notify.post_webhook("", "hi")[0] is False


# ------------------------------------------------------------------ MITRE
def test_mitre_scenarios_map_to_known_techniques_and_urls():
    for tids in mitre.SCENARIO_MAP.values():
        assert tids and all(t in mitre.TECHNIQUES for t in tids)
    for tids in mitre.FEATURE_MAP.values():
        assert all(t in mitre.TECHNIQUES for t in tids)
    assert mitre.technique_url("T1052.001") == "https://attack.mitre.org/techniques/T1052/001/"


def test_for_drivers_ranks_by_evidence_and_ignores_small_z():
    dev = pd.DataFrame({"feature": ["external_email_ratio", "max_external_size_kb", "copy_count", "avg_logon_hour"],
                        "z_score": [6.0, 4.0, 1.2, -0.5]})
    m = mitre.for_drivers(dev)
    assert m.technique.tolist() == ["T1048"] and m.evidence_score.iloc[0] == 10.0
    assert mitre.from_driver_text("copy_count (+9.0 sd), device_connect_count (+3 sd)") == ["T1005", "T1052.001"]


def test_mitre_matches_most_known_insiders(data):
    feats, labels = data["features"], data["labels"]
    cols = [c for c in feats.columns if c not in ("user_id", "is_malicious", "scenario", "injected_day")]
    bad = labels[labels.is_malicious.astype(bool)].user_id.tolist()
    m = mitre.user_matches(feats, bad, cols, soc.feature_deviations)
    hit, tot = mitre.agreement(m, labels)
    assert tot == len(bad) and hit / tot >= 0.5


# ------------------------------------------------------------------ app flows
def _app(monkeypatch, tmp_path, auth_on=True):
    monkeypatch.setenv("ITD_AUTH", "on" if auth_on else "off")
    monkeypatch.setenv("ITD_DB", str(tmp_path / "app.db"))
    return AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=240)


def _login(at, user, pw):
    at.text_input[0].set_value(user)
    at.text_input[1].set_value(pw)
    at.button[0].click()
    return at.run()


def test_login_wall_blocks_until_signed_in(monkeypatch, tmp_path):
    at = _app(monkeypatch, tmp_path).run()
    assert not at.exception and len(at.tabs) == 0 and "Sign in" in str(at.button[0].label)
    at = _login(at, "analyst", "wrong")
    assert len(at.tabs) == 0 and any("Wrong username" in e.value for e in at.error)
    at = _login(at, "analyst", "analyst123")
    assert not at.exception and len(at.tabs) == 12
    assert any(a == "login" for a in cs.audit_frame(path=str(tmp_path / "app.db")).action)


def test_viewer_is_read_only_and_analyst_can_save_and_it_persists(monkeypatch, tmp_path):
    db = str(tmp_path / "app.db")
    at = _login(_app(monkeypatch, tmp_path).run(), "viewer", "viewer123")
    save = [b for b in at.button if b.label == "Save case update"][0]
    assert save.disabled
    dl = [d for d in at.get("download_button") if "Download CSV" in str(d.proto.label)]
    assert dl and dl[0].proto.disabled

    at = _login(_app(monkeypatch, tmp_path).run(), "analyst", "analyst123")
    save = [b for b in at.button if b.label == "Save case update"][0]
    assert not save.disabled
    sb = [s for s in at.selectbox if s.label == "Status"][0]
    sb.set_value("Escalated")
    [t for t in at.text_input if t.label == "Analyst note"][0].set_value("seen exfil pattern")
    save.click()
    at.run()
    assert not at.exception
    store = cs.load_store(db)  # written to disk, analyst name recorded
    uid, case = next(iter(store.items()))
    assert case["status"] == "Escalated" and case["history"][-1]["analyst"] == "analyst"
    assert "case_update" in cs.audit_frame(path=db).action.tolist()
    assert len(cs.notifications_frame(path=db)) == 1  # escalation notified

    # a fresh session (e.g. page refresh) sees the saved case
    at2 = _login(_app(monkeypatch, tmp_path).run(), "viewer", "viewer123")
    assert at2.session_state["cases"][uid]["status"] == "Escalated"


def test_audit_log_is_admin_only(monkeypatch, tmp_path):
    at = _login(_app(monkeypatch, tmp_path).run(), "analyst", "analyst123")
    assert any("visible to administrators" in i.value for i in at.info)
    at = _login(_app(monkeypatch, tmp_path).run(), "admin", "admin123")
    assert any(b.label == "Send test notification" for b in at.button)


def test_attack_tab_and_alert_column_render():
    os.environ["ITD_AUTH"] = "off"
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=240).run()
    assert not at.exception
    assert any("Mapping check vs answer key" in m.label for m in at.metric)
