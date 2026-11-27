"""
Day 7 - AI-Powered Insider Threat Detection: SOC-style Streamlit dashboard.

Ties together every previous day's work:
  - Day 1-2: the simulated organizational activity logs
  - Day 3: engineered per-user behavioral features
  - Day 4-5: Isolation Forest / Autoencoder / Mahalanobis anomaly scores
  - Day 5: the user-resource interaction graph
  - Day 6: SHAP / LIME explanations for any selected alert

Run with:  streamlit run app.py

All the actual data loading / scoring / plotting logic lives in
src/dashboard_utils.py so it can be tested independently of Streamlit
(see tests/smoke_test_dashboard.py and docs/day7_progress.md for why
that separation mattered in this project's dev environment).
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import uuid

import pandas as pd
import streamlit as st

sys.path.insert(0, "src")
import dashboard_utils as du
import live_demo as ld
import network_view as nv
import upload_analysis as ua
import soc_tools as soc
import auth
import case_store as cs
import mitre
import notify
from realtime.scorer import RealTimeScorer, risk_from_scores
from inject_anomalies import SCENARIOS
import ui_theme
import charts

ROOT = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Insider Threat Detection | SOC Console", page_icon="🛡️", layout="wide")
st.markdown(ui_theme.CSS, unsafe_allow_html=True)
ui_theme.apply_matplotlib_theme()


def _secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:  # no secrets file / key missing
        return default


def _configured_users():
    users = auth.normalise_users(_secret("users") and {k: dict(v) for k, v in dict(_secret("users")).items()})
    return users or auth.demo_users()


def _db(fn, *args, **kwargs):
    """Run a persistence call; a read-only or broken disk must never take the dashboard down."""
    try:
        return fn(*args, **kwargs)
    except Exception:
        st.session_state["db_ok"] = False
        return None


def _login_gate():
    if not auth.auth_enabled():
        st.session_state.setdefault("user", {"name": "local", "role": "admin"})
        return
    if "user" in st.session_state:
        return
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown('<div class="brand"><div class="logo">IT</div><div><div class="name">Insider Threat</div>'
                    '<div class="sub">Detection console - sign in</div></div></div>', unsafe_allow_html=True)
        failed = st.session_state.get("failed_logins", 0)
        with st.form("login"):
            name = st.text_input("Username")
            pw = st.text_input("Password", type="password")
            go = st.form_submit_button("Sign in", type="primary", width="stretch")
        if go:
            if failed >= auth.MAX_FAILED:
                st.error("Too many failed attempts in this session. Reload the page to try again.")
            else:
                role = auth.authenticate(name.strip(), pw, _configured_users())
                if role:
                    st.session_state["user"] = {"name": name.strip(), "role": role}
                    st.session_state["failed_logins"] = 0
                    _db(cs.log_audit, name.strip(), role, "login", "success")
                    st.rerun()
                st.session_state["failed_logins"] = failed + 1
                _db(cs.log_audit, name.strip() or "?", "-", "login_failed", f"attempt {failed + 1}")
                st.error("Wrong username or password.")
        if auth.demo_logins_visible() and not _secret("users"):
            st.info("**Demo accounts** - admin / admin123, analyst / analyst123, viewer / viewer123. "
                    "Roles: *viewer* is read-only, *analyst* can work cases and export, *admin* also sees the audit log.")
    st.stop()


_login_gate()
USER = st.session_state["user"]["name"]
ROLE = st.session_state["user"]["role"]


def _can(action):
    return auth.can(ROLE, action)


def _audit(action, detail=""):
    _db(cs.log_audit, USER, ROLE, action, detail)


@st.cache_data
def _load_data():
    return du.load_data()


@st.cache_data
def _feature_matrix(_features):
    return du.get_feature_matrix(_features)


@st.cache_data(show_spinner=False)
def _network_table(model_name, min_w, _models, _X, _features, _labels, _data):
    _, risk = du.score_users(_models, model_name, _X)
    G = _full_graph(_data)
    tbl = nv.node_table(G, dict(zip(_features.user_id, risk)), _labels, min_w=min_w)
    return tbl, nv.graph_stats(G, tbl, min_w=min_w)


@st.cache_resource
def _trained_models(_X):
    return du.LazyModels(_X)  # each model trains on first use only


@st.cache_resource
def _full_graph(_data):
    return du.build_full_graph(_data)


@st.cache_data(show_spinner=False)
def _alert_table(model_name, top_k, _features, _labels, _models, _X):
    return du.build_alert_table(_features, _labels, _models, model_name, _X, top_k=top_k)


@st.cache_data(show_spinner=False)
def _explain(model_name, uid, _models, _X, _feature_cols, _user_ids):
    return du.explain_user(_models, model_name, _X, _feature_cols, _user_ids, uid, engine="library")


@st.cache_resource(show_spinner=False)
def _neighbourhood_fig(uid, _data):
    return du.subgraph_figure(_full_graph(_data), uid, radius=1)


data = _load_data()
X, feature_cols, train_mean, train_std = _feature_matrix(data["features"])
models = _trained_models(X)
user_ids = data["features"].user_id.values

st.title("AI-Powered Insider Threat Detection")

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="logo">IT</div><div><div class="name">Insider Threat</div>'
        '<div class="sub">Detection console</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"**{USER}** &nbsp; <span class='chip'>{ROLE}</span>", unsafe_allow_html=True)
    if auth.auth_enabled() and st.button("Sign out", key="logout", width="stretch"):
        _audit("logout")
        for k in ("user", "cases", "rt"):
            st.session_state.pop(k, None)
        st.rerun()
    st.divider()
    st.subheader("Detection settings")
    model_name = st.selectbox("Detection model", du.MODEL_CHOICES, index=0)
    top_k = st.slider("Number of alerts to show", min_value=5, max_value=30, value=15)
    st.divider()
    st.markdown("**Severity scale**")
    st.markdown(
        '<div class="legend">'
        + ui_theme.severity_chip("High") + " &ge; 70 &nbsp; "
        + ui_theme.severity_chip("Medium") + " 40-69 &nbsp; "
        + ui_theme.severity_chip("Low") + " &lt; 40</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption(
        "Ground-truth columns (`is_malicious`, `scenario`) come from the synthetic dataset's answer key "
        "and exist only to validate the models. A production deployment would not have them."
    )


if "cases" not in st.session_state:
    st.session_state["cases"] = _db(cs.load_store) or soc.new_case_store()


def _case_panel(uid, key):
    """Status + analyst note for one entity; every save is appended to an audit trail."""
    store = st.session_state["cases"]
    cur = soc.case_status(store, uid)
    st.markdown(f"**Case for {uid}** &nbsp; current status: `{cur}`")
    c1, c2 = st.columns([1, 2])
    writable = _can("case_write")
    new_status = c1.selectbox("Status", soc.STATUSES, index=soc.STATUSES.index(cur), key=f"{key}_status_{uid}", disabled=not writable)
    note = c2.text_input("Analyst note", key=f"{key}_note_{uid}", placeholder="What did you check? What is the outcome?", disabled=not writable)
    if not writable:
        st.caption(f"Your role ({ROLE}) is read-only. Ask an analyst or admin to update cases.")
    if st.button("Save case update", key=f"{key}_save_{uid}", disabled=not writable):
        soc.update_case(store, uid, new_status, note, analyst=USER)
        _db(cs.add_case_entry, uid, new_status, note, USER)
        _audit("case_update", f"{uid} -> {new_status}")
        _db(notify.notify_case, uid, new_status, USER, note, _secret("webhook_url"))
        st.toast(f"{uid}: {new_status}")
        st.rerun()
    hist = store.get(uid, {}).get("history", [])
    if hist:
        st.dataframe(pd.DataFrame(hist)[["time", "status", "analyst", "note"]].iloc[::-1], hide_index=True, width="stretch", height=150)


@st.cache_data(show_spinner=False)
def _attack_matches(uids):
    """ATT&CK techniques suggested by each user's strongest feature deviations (heuristic)."""
    return mitre.user_matches(data["features"], uids, feature_cols, soc.feature_deviations)


def _drivers(uid, n=3):
    dev = soc.feature_deviations(data["features"], uid, feature_cols, n)
    return ", ".join(f"{r.feature} ({r.z_score:+.1f} sd)" for r in dev.itertuples())


(tab_alerts, tab_inv, tab_rt, tab_live, tab_graph, tab_models, tab_upload,
 tab_attack, tab_notif) = st.tabs(
    ["Alert queue", "Investigate", "Real-time monitor", "Attack simulation", "Entity graph", "Model evaluation",
     "Upload & analyse", "MITRE ATT&CK", "Notifications & audit"]
)

with tab_alerts:
    alert_df = _alert_table(model_name, top_k, data["features"], data["labels"], models, X)
    st.markdown(
        ui_theme.status_strip(len(data["features"]), model_name, int((alert_df.risk_level == "High").sum())),
        unsafe_allow_html=True,
    )

    cases = st.session_state["cases"]
    alert_cases = soc.with_case_columns(alert_df, cases)
    _m = _attack_matches(tuple(alert_df.user_id))
    alert_cases["attack"] = alert_cases.user_id.map(
        lambda u: ", ".join(_m[u].technique.head(2)) if len(_m[u]) else "-")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Users monitored", len(data["features"]))
    col2.metric("Alerts shown", len(alert_df))
    col3.metric("High risk", int((alert_df.risk_level == "High").sum()))
    col4.metric("Open cases", int(alert_cases.case_status.isin(soc.OPEN_STATUSES).sum()))
    col5.metric("Confirmed insiders in queue", int(alert_df.is_malicious.sum()))

    st.subheader(f"Top {top_k} alerts - {model_name}")
    st.caption("Click a row to open it below. Case status and notes are saved to the database (see the Notifications & audit tab); export them as CSV for a durable copy.")
    sel = st.dataframe(
        ui_theme.style_risk_table(
            alert_cases[["rank", "user_id", "risk_score", "risk_level", "case_status", "attack", "is_malicious", "scenario"]]
        ),
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="alert_table",
        column_config={
            "rank": st.column_config.NumberColumn("Rank", width="small"),
            "user_id": st.column_config.TextColumn("Entity"),
            "risk_score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=100, format="%.1f"),
            "risk_level": st.column_config.TextColumn("Severity", width="small"),
            "case_status": st.column_config.TextColumn("Case status"),
            "attack": st.column_config.TextColumn("ATT&CK (suggested)"),
            "is_malicious": st.column_config.CheckboxColumn("Ground truth: insider", width="small"),
            "scenario": st.column_config.TextColumn("Injected scenario"),
        },
    )
    picked = sel.selection.rows[0] if sel.selection.rows else 0

    e1, e2, e3 = st.columns([1, 1, 2])
    _can_export = _can("export")
    e1.download_button("Download CSV", soc.alerts_csv(alert_df, cases), "alerts.csv", "text/csv", width="stretch",
                       disabled=not _can_export, on_click=_audit, args=("export", "alerts.csv"))
    try:
        import reportlab  # noqa: F401
        e2.download_button("Download PDF report",
                           soc.alerts_pdf(alert_df, cases, model_name, {u: _drivers(u) for u in alert_df.user_id}),
                           "alert_report.pdf", "application/pdf", width="stretch",
                           disabled=not _can_export, on_click=_audit, args=("export", "alert_report.pdf"))
    except ImportError:
        e2.button("PDF export unavailable", disabled=True, width="stretch", help="Install reportlab to enable PDF export.")
    with e3.expander("Save / restore case log"):
        log = soc.cases_to_frame(cases)
        st.download_button("Download case log (CSV)", log.to_csv(index=False).encode("utf-8"), "case_log.csv", "text/csv",
                           disabled=log.empty or not _can_export, on_click=_audit, args=("export", "case_log.csv"))
        up = st.file_uploader("Restore a saved case log (admin only)", type="csv", key="case_upload", disabled=not _can("admin"))
        if up is not None and st.button("Load case log", key="case_load", disabled=not _can("admin")):
            st.session_state["cases"] = soc.cases_from_frame(pd.read_csv(up))
            _db(cs.replace_cases, st.session_state["cases"])
            _audit("case_log_restored", f"{len(st.session_state['cases'])} entities")
            st.rerun()

    st.subheader("Explain an alert")
    selected_uid = st.selectbox("Select a user to explain", alert_df.user_id.tolist(), index=int(picked))

    if selected_uid:
        row = alert_df[alert_df.user_id == selected_uid].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk score", f"{row.risk_score:.1f}/100")
        c2.metric("Risk level", row.risk_level)
        c3.metric("Ground truth", "Malicious" if row.is_malicious else "Normal")

        _case_panel(selected_uid, "aq")

        with st.spinner("Computing SHAP / LIME explanation..."):
            shap_s, lime_s = _explain(model_name, selected_uid, models, X, feature_cols, user_ids)
            fig = du.explanation_figure(shap_s, lime_s, selected_uid)
        st.pyplot(fig)
        st.caption(
            "SHAP (left) shows each feature's contribution to this user's anomaly "
            "score relative to the population average. LIME (right) shows a local "
            "linear approximation around this specific user - see docs/day6_progress.md "
            "for why LIME is noisier than SHAP on this model."
        )

        st.subheader(f"{selected_uid}'s resource/relationship neighbourhood")
        subfig = _neighbourhood_fig(selected_uid, data)
        if subfig:
            st.pyplot(subfig)


def _stop_stream():
    rt = st.session_state.get("rt")
    if rt:
        try:
            open(os.path.join(rt["dir"], "stop"), "w").close()
            if rt["proc"].poll() is None:
                rt["proc"].terminate()
        except OSError:
            pass


def _start_stream(interval):
    _stop_stream()
    d = tempfile.mkdtemp(prefix="insider_live_")
    proc = subprocess.Popen(
        [sys.executable, os.path.join(ROOT, "src", "realtime", "event_producer.py"),
         "--dir", d, "--interval", str(interval), "--ticks", "80"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    model = models[model_name]
    train_scores = model.decision_function(X)
    scorer = RealTimeScorer(os.path.join(d, "events.jsonl"), model, feature_cols, train_mean, train_std,
                            train_scores.min(), train_scores.max())
    st.session_state["rt"] = {"proc": proc, "dir": d, "scorer": scorer, "done": False, "model": model_name}


with tab_inv:
    st.subheader("Investigate an entity")
    st.caption("Everything an analyst needs about one user in one place: unusual behaviours, activity over time, "
               "comparison with their department, model explanation, network context and the case record.")
    base_alerts = _alert_table(model_name, 100, data["features"], data["labels"], models, X)
    ids_by_risk = base_alerts.user_id.tolist()
    inv_uid = st.selectbox("Entity (highest risk first)", ids_by_risk, key="inv_uid")
    inv_row = base_alerts[base_alerts.user_id == inv_uid].iloc[0]
    urow = data["users"][data["users"].user_id == inv_uid].iloc[0]
    lab = data["labels"][data["labels"].user_id == inv_uid].iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Risk score", f"{inv_row.risk_score:.1f}/100")
    k2.metric("Rank", f"#{int(inv_row['rank'])} of {len(base_alerts)}")
    k3.metric("Department", urow.department)
    k4.metric("Case status", soc.case_status(st.session_state["cases"], inv_uid))
    st.caption(f"{urow.full_name} - {urow['role']} - {urow.email}")

    tl, dv = st.columns([1.15, 1])
    with tl:
        tchart = charts.timeline_chart(soc.activity_timeline(data["file_df"], data["email_df"], inv_uid),
                                       lab.injected_day if bool(lab.is_malicious) else None)
        if tchart is not None:
            st.altair_chart(tchart, width="stretch")
            if bool(lab.is_malicious):
                st.caption("Red dashed line: day the attack was injected (ground truth, demo data only).")
        else:
            st.info("No file or e-mail activity recorded for this user.")
    with dv:
        st.altair_chart(charts.deviation_chart(soc.feature_deviations(data["features"], inv_uid, feature_cols, 10)),
                        width="stretch")

    st.markdown("**Compared with the same department**")
    peer = soc.peer_comparison(data["features"], data["users"], inv_uid, feature_cols, 6)
    st.dataframe(peer.drop(columns="department").round(2), hide_index=True, width="stretch",
                 column_config={"feature": "Feature", "value": "This user", "department_mean": f"{urow.department} mean",
                                "population_mean": "Everyone", "z_score": "z-score"})
    sh, gr = st.columns(2)
    with sh:
        st.markdown("**Why the model flagged this user (SHAP / LIME)**")
        with st.spinner("Computing explanation..."):
            _s, _l = _explain(model_name, inv_uid, models, X, feature_cols, user_ids)
            st.pyplot(du.explanation_figure(_s, _l, inv_uid))
    with gr:
        st.markdown("**Resource / relationship neighbourhood**")
        _sf = _neighbourhood_fig(inv_uid, data)
        if _sf:
            st.pyplot(_sf)
    _case_panel(inv_uid, "inv")


with tab_rt:
    st.subheader("Real-time monitoring (micro-batch)")
    st.caption(
        "A separate process plays the organisation's log pipeline and appends events to a log file "
        "(one simulated working day per tick). The scorer tails that file, rebuilds each user's features "
        "over a 30-day sliding window, scores them with the selected model and raises an alert when risk "
        "crosses the threshold. Risk is on an absolute scale learned from the training population, so quiet "
        "users stay low. Speed and threshold can be changed while the stream is running."
    )
    r1, r2, r3, r4 = st.columns(4)
    rt_users = ld.create_live_cohort(n=12, seed=123).user_id.tolist()
    rt_user = r1.selectbox("Insider to attack", rt_users, index=2, key="rt_user")
    rt_scen = r2.selectbox("Attack scenario", SCENARIOS, index=1, key="rt_scen")
    rt_interval = r3.slider("Seconds per simulated day", 0.2, 5.0, 1.5, 0.1, key="rt_interval")
    rt_thr = r4.slider("Alert threshold", 40, 95, 70, 5, key="rt_thr")
    b1, b2, b3, b4 = st.columns(4)
    _sim = _can("simulate")
    if not _sim:
        st.caption(f"Your role ({ROLE}) can watch the stream but not control it.")
    if b1.button("Start stream", type="primary", key="rt_start", width="stretch", disabled=not _sim):
        _start_stream(rt_interval)
        _audit("stream_start", f"model={model_name}")
    if b2.button("Inject attack now", key="rt_inject", width="stretch", disabled=not _sim):
        _audit("attack_injected", f"{rt_user} / {rt_scen}")
        rt = st.session_state.get("rt")
        if rt and not rt["done"]:
            with open(os.path.join(rt["dir"], "control.json"), "w", encoding="utf-8") as f:
                json.dump({"nonce": uuid.uuid4().hex, "user": rt_user, "scenario": rt_scen}, f)
            st.toast(f"Attack `{rt_scen}` will be injected for {rt_user} on the next simulated day")
        else:
            st.warning("Start the stream first.")
    _rt0 = st.session_state.get("rt")
    _paused = bool(_rt0 and os.path.exists(os.path.join(_rt0["dir"], "pause")))
    if b3.button("Resume" if _paused else "Pause", key="rt_pause", width="stretch", disabled=not _sim):
        if _rt0 and not _rt0["done"]:
            pp = os.path.join(_rt0["dir"], "pause")
            if _paused:
                os.remove(pp)
            else:
                open(pp, "w").close()
            st.rerun()
    if b4.button("Stop", key="rt_stop", width="stretch", disabled=not _sim):
        _stop_stream()
        if st.session_state.get("rt"):
            st.session_state["rt"]["done"] = True

    rt_state = st.session_state.get("rt")
    rt_running = bool(rt_state and not rt_state["done"])
    if rt_running:  # live controls: speed (producer reads the file each tick) and alert threshold
        try:
            with open(os.path.join(rt_state["dir"], "interval"), "w", encoding="utf-8") as f:
                f.write(str(rt_interval))
        except OSError:
            pass
        rt_state["scorer"].threshold = float(rt_thr)
        rt_state["scorer"].rearm_below = float(rt_thr) - 20.0

    def _rt_panel():
        rt = st.session_state.get("rt")
        if not rt:
            st.info("Press **Start stream**, wait a few ticks, then press **Inject attack now** and watch the risk and the alert.")
            return
        scorer = rt["scorer"]
        state = scorer.poll()
        if state is None and rt["proc"].poll() is not None and not rt["done"]:
            rt["done"] = True
            st.rerun()  # full rerun so the auto-refresh stops
        state = scorer.last_state
        if state is None:
            st.info("Waiting for the first events...")
            return
        alerts = scorer.alerts
        sent = rt.setdefault("notified", set())
        for a in alerts:
            k = (a["tick"], a["user_id"])
            if k not in sent:
                sent.add(k)
                tech = ", ".join(mitre.technique_label(t) for t in mitre.from_driver_text(a.get("drivers", ""))[:2])
                _db(notify.notify_alert, dict(a, stream=os.path.basename(rt["dir"])), tech, _secret("webhook_url"))
        truth_users = {t["user"] for t in scorer.truth}
        paused = os.path.exists(os.path.join(rt["dir"], "pause"))
        status = "STREAM ENDED" if rt["done"] else ("PAUSED" if paused else "STREAMING")
        st.markdown(f"**Status: {status}** &nbsp;|&nbsp; model: {rt['model']} &nbsp;|&nbsp; threshold: {scorer.threshold:.0f}")
        lat = scorer.latency_stats()
        m = st.columns(6)
        m[0].metric("Tick", state["tick"])
        m[1].metric("Simulated day", pd.Timestamp(state["sim_date"]).strftime("%d %b"))
        m[2].metric("Events ingested", f"{scorer.n_events:,}")
        m[3].metric("Throughput", f"{scorer.throughput():.0f} ev/s")
        m[4].metric("Active alerts", state.get("active_alerts", 0))
        m[5].metric("Latency p95", "-" if lat["p50"] is None else f"{lat['p95']:.2f} s",
                    help="Time from the producer writing a batch to the scorer finishing it. "
                         + ("" if lat["p50"] is None else f"Median {lat['p50']:.2f} s, worst {lat['max']:.2f} s."))

        hist = scorer.history_df()
        top = state["risk_table"].user_id.head(4).tolist()
        cols = list(dict.fromkeys(top + ([rt_user] if rt_user in hist.columns else [])))
        left, right = st.columns([2, 1])
        left.altair_chart(charts.risk_timeseries(hist[cols], rt_user, alerts), width="stretch")
        right.altair_chart(charts.live_leaderboard(state["risk_table"], scorer.threshold), width="stretch")
        hl, hr = st.columns([2, 1])
        hm = charts.risk_heatmap(hist)
        if hm is not None:
            hl.altair_chart(hm, width="stretch")
        lc = charts.latency_chart(scorer.tick_stats)
        if lc is not None:
            hr.altair_chart(lc, width="stretch")

        for a in alerts[-3:][::-1]:
            hit = a["user_id"] in truth_users
            st.error(f"ALERT: {a['user_id']} flagged HIGH RISK on simulated day {a['tick']} (risk {a['risk_score']:.0f}/100, "
                     f"detected in {a['latency_s']:.2f} s) - "
                     + ("matches the injected attack" if hit else "no injected attack for this user (possible false alarm)"))

        al, fd = st.columns([1.3, 1])
        with al:
            st.markdown("**Alert log**")
            if alerts:
                log = pd.DataFrame([{"Day": a["sim_date"], "Tick": a["tick"], "User": a["user_id"], "Risk": round(a["risk_score"]),
                                     "Latency (s)": a["latency_s"], "Top drivers": a.get("drivers", ""),
                                     "Injected attack": "yes" if a["user_id"] in truth_users else "no (false alarm)"}
                                    for a in reversed(alerts)])
                st.dataframe(log, hide_index=True, width="stretch", height=240)
            else:
                st.caption("No alerts yet.")
        with fd:
            st.markdown("**Live event feed (newest first)**")
            st.dataframe(scorer.feed_df(15), hide_index=True, width="stretch", height=240)

    st.fragment(run_every=1.0 if rt_running else None)(_rt_panel)()

with tab_live:
    st.subheader("Live feed simulation")
    st.caption(
        "Simulates a small demo company generating fresh activity day by day, "
        "scored in real time against the SAME model trained on the Day 1-6 "
        "baseline population - exactly like a real deployed system would score "
        "new incoming behaviour. Pick a scenario, start the feed, and watch the "
        "risk score for the chosen user jump the moment the attack is injected."
    )

    live_cohort = ld.create_live_cohort(n=12, seed=123)

    lc1, lc2, lc3, lc4 = st.columns(4)
    demo_user_id = lc1.selectbox("Demo user (the 'insider')", live_cohort.user_id.tolist(), index=2)
    scenario = lc2.selectbox("Attack scenario to inject", SCENARIOS, index=1)
    n_ticks = lc3.slider("Simulated days to run", min_value=6, max_value=20, value=10)
    inject_at_tick = lc4.slider("Inject attack on day", min_value=2, max_value=n_ticks, value=min(6, n_ticks))
    speed = st.slider("Seconds per simulated day (demo pacing)", min_value=0.2, max_value=3.0, value=1.0, step=0.2)

    start = st.button("Start Live Feed", type="primary", disabled=not _can("simulate"))

    if start:
        live_model = models[model_name]
        metrics_placeholder = st.empty()
        chart_placeholder = st.empty()
        table_placeholder = st.empty()
        alert_placeholder = st.empty()

        history = []
        gen = ld.run_live_demo(
            live_cohort, feature_cols, train_mean, train_std, live_model,
            n_ticks=n_ticks, inject_at_tick=inject_at_tick,
            demo_user_id=demo_user_id, scenario=scenario,
        )

        for state in gen:
            demo_row = state["risk_table"][state["risk_table"].user_id == demo_user_id].iloc[0]
            history.append({"tick": state["tick"], "demo_user_risk": demo_row.risk_score})

            with metrics_placeholder.container():
                m1, m2, m3 = st.columns(3)
                m1.metric("Simulated day", f"{state['tick']} / {n_ticks}")
                m2.metric(f"{demo_user_id} risk score", f"{demo_row.risk_score:.1f}/100")
                m3.metric("Status", "ATTACK ACTIVE" if state["tick"] >= inject_at_tick else "monitoring...")

            chart_placeholder.altair_chart(charts.live_risk_chart(history, demo_user_id, inject_at_tick), width="stretch")

            table_placeholder.dataframe(
                state["risk_table"].head(8).rename(
                    columns={"user_id": "User", "anomaly_score": "Raw score", "risk_score": "Risk (0-100)"}
                ),
                width="stretch",
                hide_index=True,
            )

            if state["injected"] or (state["tick"] > inject_at_tick and demo_row.risk_score >= 70):
                alert_placeholder.error(
                    f"ALERT: {demo_user_id} flagged HIGH RISK on simulated day {state['tick']} "
                    f"(risk score {demo_row.risk_score:.1f}/100) - injected scenario: `{scenario}`"
                )

            time.sleep(speed)

        st.success(
            f"Live feed complete. {demo_user_id}'s risk score jumped from the "
            f"pre-attack range straight to {history[-1]['demo_user_risk']:.1f}/100 "
            f"the moment the `{scenario}` scenario was injected on day {inject_at_tick}, "
            "and stayed flagged for every day afterward."
        )

with tab_graph:
    st.subheader("Organisation network - users, departments and shared folders")
    st.caption(
        "Deterministic cluster layout (no animation): departments on a ring, shared folders in the centre. "
        "Colour = risk (light blue low, dark blue high); size = e-mail strength; black outline = ground-truth insider "
        "(demo data only). Scroll to zoom, drag to pan, hover for metrics."
    )
    c1, c2, c3 = st.columns([1.2, 1.2, 1.6])
    with c1:
        min_links = st.slider("Minimum e-mails to draw a link", 1, 15, 7,
                              help="Lower = more links. 7 keeps the strongest ~5% of user-user links.")
    with c2:
        label_thr = st.slider("Label users with risk at or above", 0, 100, 50)
    focus_options = ["(whole organisation)"] + data["features"].user_id.tolist()
    with c3:
        focus_choice = st.selectbox("Focus on one user and their direct contacts", focus_options, index=0)
    focus_user = None if focus_choice == focus_options[0] else focus_choice
    tbl, gstats = _network_table(model_name, min_links, models, X, data["features"], data["labels"], data)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Users", gstats["nodes"])
    k2.metric("Links shown", gstats["edges"])
    k3.metric("Density", f"{gstats['density']:.3f}")
    k4.metric("Components", gstats["components"])
    k5.metric("Dept modularity", f"{gstats['modularity']:.2f}",
              help="Near 0 = e-mail traffic does not follow department boundaries.")
    st.altair_chart(nv.network_chart(_full_graph(data), tbl, min_w=min_links, label_risk=label_thr, focus=focus_user),
                    width="stretch")
    left, right = st.columns([1.1, 1])
    with left:
        st.altair_chart(nv.risk_vs_strength_chart(tbl, gstats["spearman_risk_strength"]), width="stretch")
        st.caption("Risk is not explained by how much a user e-mails (rho near 0): the models flag behaviour, "
                   "not network position - which is why the graph is context for an analyst, not a detector.")
    with right:
        st.markdown("**Top users by risk - network metrics**")
        top = tbl.sort_values("risk", ascending=False).head(10)[["user", "department", "risk", "strength", "betweenness", "scenario"]]
        st.dataframe(top.round(3), hide_index=True, width="stretch")
    with st.expander("Static graph + centrality note"):
        st.image(
            "docs/figures/user_resource_graph.png",
            caption="Red = known malicious users, blue = normal users, yellow squares = shared resources",
            width="stretch",
        )
        st.caption(
            "See docs/day5_progress.md - graph centrality alone does not separate "
            "malicious users in the current dataset; documented there as an honest limitation."
        )

with tab_models:
    st.subheader("Do the models agree?")
    st.caption("All detectors score the same 100 users. Where they agree the alert is more trustworthy; "
               "an entity flagged by one model only deserves a second look before it consumes analyst time.")
    cmp_k = st.slider("Compare each model's top-k alerts", 5, 30, 10, key="cmp_k")
    with st.spinner("Scoring with every model..."):
        ranks, overlap, consensus = soc.model_rankings(lambda m: du.score_users(models, m, X)[1], du.MODEL_CHOICES,
                                                       list(user_ids), cmp_k)
    truth = data["labels"].set_index("user_id").is_malicious.to_dict()
    ca, cb = st.columns([1.1, 1])
    with ca:
        _short = lambda n: n.replace("Isolation Forest", "IForest").replace(" (sklearn)", " (sk)").replace(" (Keras)", " (K)").replace(" detector", "")
        st.altair_chart(charts.agreement_heatmap(overlap.rename(index=_short, columns=_short)), width="stretch")
    with cb:
        st.markdown("**Consensus alerts**")
        cons = consensus.copy()
        cons["ground_truth_insider"] = cons.user_id.map(truth).fillna(False)
        cons["mean_rank"] = cons.mean_rank.round(1)
        st.dataframe(cons.head(12), hide_index=True, width="stretch",
                     column_config={"models_flagging": st.column_config.ProgressColumn(
                         f"Models flagging (of {len(du.MODEL_CHOICES)})", min_value=0, max_value=len(du.MODEL_CHOICES), format="%d"),
                         "ground_truth_insider": st.column_config.CheckboxColumn("Ground truth")})
    side = {m: ranks[m].nsmallest(cmp_k).index.tolist() for m in du.MODEL_CHOICES}
    st.markdown("**Side-by-side top alerts**")
    st.dataframe(pd.DataFrame(side, index=[f"#{i}" for i in range(1, cmp_k + 1)]), width="stretch")

    st.subheader("Alert threshold simulator")
    st.caption("Move the threshold and see the trade-off: a low threshold catches every insider but floods the analysts with "
               "false alarms; a high one is quiet but misses attacks. Uses the dataset's ground truth, so it is a tuning aid "
               "for the demo data, not something a production deployment could compute directly.")
    sim_model = st.selectbox("Model", du.MODEL_CHOICES, index=du.MODEL_CHOICES.index(model_name), key="sim_model")
    sim_thr = st.slider("Alert threshold", 0, 100, 70, 1, key="sim_thr")
    risk_all = du.score_users(models, sim_model, X)[1]
    y_all = data["labels"].set_index("user_id").loc[list(user_ids), "is_malicious"].astype(bool).values
    sweep = soc.threshold_sweep(risk_all, y_all)
    at_t = sweep[sweep.threshold == sim_thr].iloc[0]
    best = soc.best_threshold(sweep)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Users alerted", int(at_t.alerts))
    m2.metric("Precision", f"{at_t.precision:.0%}")
    m3.metric("Recall", f"{at_t.recall:.0%}")
    m4.metric("False alarms", int(at_t.false_positives))
    m5.metric("Insiders missed", int(at_t.missed_insiders))
    st.caption(f"Best F1 on this dataset: threshold {best.threshold:.0f} (precision {best.precision:.0%}, recall {best.recall:.0%}).")
    s1, s2 = st.columns(2)
    s1.altair_chart(charts.threshold_chart(sweep, sim_thr), width="stretch")
    s2.altair_chart(charts.alert_volume_chart(sweep, sim_thr), width="stretch")

    st.subheader("Model comparison (Day 4 + Day 5)")
    st.dataframe(data["model_comparison"].round(3), width="stretch")
    col1, col2 = st.columns(2)
    with col1:
        st.image("docs/figures/day4_roc_curves.png", caption="Day 4 ROC curves", width="stretch")
    with col2:
        st.image("docs/figures/day5_autoencoder.png", caption="Day 5 Autoencoder training + ROC", width="stretch")
    st.caption(
        "All models achieve near-perfect separation on this single 100-user dataset - read "
        "docs/day4_progress.md's caveat, and see the multi-seed evaluation below for a realistic picture."
    )
    st.subheader("Robust evaluation (8 unseen datasets, 200 users, 10 insiders each)")
    try:
        robust = pd.read_csv("data/processed/full/robust_evaluation.csv")
        summary = robust.groupby("model")[["roc_auc", "precision@k", "recall@k"]].agg(["mean", "std"]).round(3)
        summary.columns = [f"{a} ({b})" for a, b in summary.columns]
        st.dataframe(summary.sort_values("roc_auc (mean)", ascending=False), width="stretch")
        st.image("docs/figures/robust_evaluation.png", width="stretch")
        st.caption("Right panel: detection quality as the injected attack is weakened (low-and-slow insiders). See docs/robust_evaluation.md.")
    except FileNotFoundError:
        st.info("Run `python3 src/robust_evaluation.py` to generate the multi-seed evaluation.")


@st.cache_data(show_spinner=False, max_entries=4)
def _ingest_upload(payload):
    """payload: tuple of (file name, bytes). Cached so re-runs do not re-parse the files."""
    return ua.prepare({n: b for n, b in payload})


@st.cache_data(show_spinner=False)
def _sample_bytes(name):
    with open(os.path.join(ROOT, du.DATA_DIR, name), "rb") as f:
        return f.read()


with tab_upload:
    st.subheader("Analyse your own log files")
    st.caption(
        "Upload CSV logs and get the same risk ranking and explanations as the demo dataset. Nothing is stored: "
        "files are processed in memory for this session only. Do not upload real personal data to a public demo."
    )
    with st.expander("File format and sample files", expanded=False):
        st.markdown(
            "- **Raw logs** (any of them): `logon.csv` (user_id, pc, activity, timestamp), `device.csv` (user_id, pc, activity, timestamp), "
            "`http.csv` (user_id, pc, url, timestamp), `email.csv` (user_id, to, external, size_kb, attachments, timestamp), "
            "`file.csv` (user_id, pc, filename, action, timestamp); optional `users.csv` (user_id, ...).\n"
            "- **or one feature table:** `features.csv` with `user_id` plus the 28 feature columns.\n"
            f"- At least **{ua.MIN_USERS} users** are needed so that normal behaviour can be learned. "
            "File names must contain the log type (e.g. `my_logon_week1.csv`)."
        )
        cols = st.columns(6)
        for col, fname in zip(cols, ["logon.csv", "device.csv", "http.csv", "email.csv", "file.csv", "users.csv"]):
            col.download_button(f"Sample {fname}", _sample_bytes(fname), file_name=fname, mime="text/csv", key=f"dl_{fname}")
        st.download_button("Sample features.csv (28 columns)", _sample_bytes("user_features.csv"),
                           file_name="features.csv", mime="text/csv", key="dl_features")
    uploads = st.file_uploader("Drop CSV files here (you can select several)", type=["csv"], accept_multiple_files=True)
    up_model = st.selectbox("Detection model for uploaded data", du.MODEL_CHOICES, index=0, key="up_model")
    if not uploads:
        st.info("No files uploaded yet. Download the sample files above, change them if you like, and upload them here.")
    else:
        try:
            up_feats, up_msgs = _ingest_upload(tuple((f.name, f.getvalue()) for f in uploads))
        except ValueError as err:
            st.error(str(err))
            up_feats = None
        if up_feats is not None:
            for m in up_msgs:
                st.warning(m)
            up_X, up_cols, _, _ = du.get_feature_matrix(up_feats)
            up_models = du.LazyModels(up_X)
            with st.spinner(f"Scoring {len(up_feats)} users with {up_model}..."):
                res = ua.results_table(up_feats, up_models, up_model, up_X, du)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Users analysed", len(res))
            k2.metric("High risk", int((res.risk_level == "High").sum()))
            k3.metric("Medium risk", int((res.risk_level == "Medium").sum()))
            k4.metric("Highest risk", f"{res.risk_score.max():.1f}")
            st.altair_chart(charts.risk_distribution(res), width="stretch")
            st.markdown("**Ranked users**")
            st.dataframe(
                ui_theme.style_risk_table(res.head(30)[["rank", "user_id", "risk_score", "risk_level"]]),
                hide_index=True, width="stretch",
                column_config={"risk_score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=100, format="%.1f")},
            )
            st.download_button("Download full results (CSV)", res.to_csv(index=False).encode(),
                               file_name="risk_results.csv", mime="text/csv")
            up_uid = st.selectbox("Why was this user flagged?", res.user_id.head(30).tolist(), key="up_uid")
            drivers = ua.top_drivers(up_feats, up_X, up_cols, up_uid)
            st.markdown(
                "**Strongest deviations from the population (z-score):** "
                + ", ".join(f"`{k}` ({v:+.1f} s.d.)" for k, v in drivers.items())
            )
            if st.checkbox("Compute full SHAP / LIME explanation (slower)", key="up_shap"):
                with st.spinner("Computing SHAP / LIME..."):
                    s_, l_ = du.explain_user(up_models, up_model, up_X, up_cols, up_feats.user_id.values, up_uid, engine="library")
                    st.pyplot(du.explanation_figure(s_, l_, up_uid))


# ------------------------------------------------------------------ MITRE ATT&CK
with tab_attack:
    st.subheader("MITRE ATT&CK view")
    st.caption(
        "Each alert's strongest behavioural deviations are mapped to MITRE ATT&CK techniques as investigation hints. "
        "The mapping is heuristic (feature -> technique) and is *not* proof of a technique being used. "
        "It is built from behaviour only, never from the ground-truth columns."
    )
    atk_df = _alert_table(model_name, top_k, data["features"], data["labels"], models, X)
    atk_m = _attack_matches(tuple(atk_df.user_id))
    cov = mitre.coverage_matrix({u: m.technique.tolist() for u, m in atk_m.items()})
    hit, tot = mitre.agreement(atk_m, data["labels"])
    a1, a2, a3 = st.columns(3)
    a1.metric("Alerts analysed", len(atk_df))
    a2.metric("Techniques observed", int((cov.users_flagged > 0).sum()))
    a3.metric("Mapping check vs answer key", f"{hit}/{tot}" if tot else "-",
              help="Of the known insiders in this queue, how many have a suggested technique that belongs to their injected scenario. "
                   "Validation only: the answer key does not exist in production.")
    st.markdown("**Coverage: users flagged per technique**")
    st.dataframe(cov.assign(technique=cov.technique.map(mitre.technique_url)), hide_index=True, width="stretch",
                 column_config={"tactic": "Tactic",
                                "technique": st.column_config.LinkColumn("Technique", display_text=r"attack\.mitre\.org/techniques/(.*)/"),
                                "name": "Name", "users_flagged": st.column_config.ProgressColumn("Users flagged", min_value=0, max_value=max(1, int(cov.users_flagged.max())), format="%d"),
                                "users": "Users"})
    st.markdown("**One user's evidence**")
    pick = st.selectbox("User", atk_df.user_id.tolist(), key="atk_user")
    ev = atk_m[pick]
    if len(ev):
        st.dataframe(ev.rename(columns={"technique": "Technique", "name": "Name", "tactic": "Tactic",
                                        "evidence_score": "Evidence (sum |z|)", "features": "Supporting features"}),
                     hide_index=True, width="stretch")
    else:
        st.info("No feature deviates enough (|z| >= 2) to suggest a technique for this user.")
    with st.expander("Scenario reference: injected scenario -> ATT&CK"):
        st.dataframe(pd.DataFrame([{"Scenario": k, "Techniques": ", ".join(mitre.technique_label(t) for t in v)}
                                   for k, v in mitre.SCENARIO_MAP.items()]), hide_index=True, width="stretch")


# ------------------------------------------------------------------ notifications and audit
with tab_notif:
    st.subheader("Notifications")
    hook = notify.webhook_url(_secret("webhook_url"))
    st.caption("Real-time alerts and escalated cases are posted to this feed. "
               + ("A webhook is configured: alerts are also pushed to it." if hook else
                  "No webhook is configured (set `webhook_url` in Streamlit secrets or ITD_WEBHOOK_URL to also push to Slack/Teams-style channels)."))
    if st.session_state.get("db_ok") is False:
        st.warning("The database could not be written on this machine, so cases and notifications are session-only.")
    feed = _db(cs.notifications_frame)
    n1, n2 = st.columns([3, 1])
    n2.metric("Notifications", 0 if feed is None else len(feed))
    if feed is not None and len(feed):
        n1.dataframe(feed.rename(columns={"time": "Time", "severity": "Severity", "title": "Title", "message": "Details",
                                          "user_id": "Entity", "source": "Source", "delivered": "Delivered"}),
                     hide_index=True, width="stretch", height=260)
        st.download_button("Download notifications (CSV)", feed.to_csv(index=False).encode("utf-8"), "notifications.csv", "text/csv",
                           disabled=not _can("export"), on_click=_audit, args=("export", "notifications.csv"))
    else:
        n1.info("No notifications yet. Start the Real-time stream and inject an attack, or escalate a case.")

    st.subheader("Audit log")
    if not _can("admin"):
        st.info("The audit log (sign-ins, case changes, exports, stream control) is visible to administrators.")
    else:
        t1, t2 = st.columns(2)
        if t1.button("Send test notification", key="notif_test"):
            ok = _db(cs.add_notification, "Test notification", f"Sent by {USER}", "Low", "", "admin", "feed",
                     f"test:{cs._now()}")
            if hook:
                ok_h, info = notify.post_webhook(hook, f"Test notification from the Insider Threat console ({USER})")
                st.toast(f"Webhook: {info}")
            _audit("notification_test")
            st.rerun()
        if t2.button("Clear notification feed", key="notif_clear"):
            _db(cs.clear_notifications)
            _audit("notifications_cleared")
            st.rerun()
        audit = _db(cs.audit_frame)
        if audit is not None and len(audit):
            st.dataframe(audit, hide_index=True, width="stretch", height=300)
            st.download_button("Download audit log (CSV)", audit.to_csv(index=False).encode("utf-8"), "audit_log.csv", "text/csv",
                               on_click=_audit, args=("export", "audit_log.csv"))
        else:
            st.caption("No audit entries yet.")
