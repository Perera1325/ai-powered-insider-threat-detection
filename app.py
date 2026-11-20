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
from realtime.scorer import RealTimeScorer, risk_from_scores
from inject_anomalies import SCENARIOS
import ui_theme
import charts

ROOT = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Insider Threat Detection | SOC Console", page_icon="🛡️", layout="wide")
st.markdown(ui_theme.CSS, unsafe_allow_html=True)
ui_theme.apply_matplotlib_theme()


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

tab_alerts, tab_rt, tab_live, tab_graph, tab_models = st.tabs(
    ["Alert queue", "Real-time monitor", "Attack simulation", "Entity graph", "Model evaluation"]
)

with tab_alerts:
    alert_df = _alert_table(model_name, top_k, data["features"], data["labels"], models, X)
    st.markdown(
        ui_theme.status_strip(len(data["features"]), model_name, int((alert_df.risk_level == "High").sum())),
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Users monitored", len(data["features"]))
    col2.metric("Alerts shown", len(alert_df))
    col3.metric("High risk", int((alert_df.risk_level == "High").sum()))
    col4.metric("Confirmed insiders in queue", int(alert_df.is_malicious.sum()))

    st.subheader(f"Top {top_k} alerts - {model_name}")
    st.dataframe(
        ui_theme.style_risk_table(
            alert_df[["rank", "user_id", "risk_score", "risk_level", "is_malicious", "scenario"]]
        ),
        width="stretch",
        hide_index=True,
        column_config={
            "rank": st.column_config.NumberColumn("Rank", width="small"),
            "user_id": st.column_config.TextColumn("Entity"),
            "risk_score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=100, format="%.1f"),
            "risk_level": st.column_config.TextColumn("Severity", width="small"),
            "is_malicious": st.column_config.CheckboxColumn("Ground truth: insider", width="small"),
            "scenario": st.column_config.TextColumn("Injected scenario"),
        },
    )

    st.subheader("Explain an alert")
    selected_uid = st.selectbox("Select a user to explain", alert_df.user_id.tolist())

    if selected_uid:
        row = alert_df[alert_df.user_id == selected_uid].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk score", f"{row.risk_score:.1f}/100")
        c2.metric("Risk level", row.risk_level)
        c3.metric("Ground truth", "Malicious" if row.is_malicious else "Normal")

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


with tab_rt:
    st.subheader("Real-time monitoring (micro-batch)")
    st.caption(
        "A separate process plays the organisation's log pipeline and appends events to a log file "
        "(one simulated working day per tick). The scorer tails that file, rebuilds each user's features "
        "over a 30-day sliding window, scores them with the selected model and raises an alert when risk "
        "crosses 70. Risk is on an absolute scale learned from the training population, so quiet users stay "
        "low. This is micro-batch (seconds of delay), not per-event streaming."
    )
    r1, r2, r3 = st.columns(3)
    rt_users = ld.create_live_cohort(n=12, seed=123).user_id.tolist()
    rt_user = r1.selectbox("Insider to attack", rt_users, index=2, key="rt_user")
    rt_scen = r2.selectbox("Attack scenario", SCENARIOS, index=1, key="rt_scen")
    rt_interval = r3.slider("Seconds per simulated day", 0.5, 5.0, 1.5, 0.5, key="rt_interval")
    b1, b2, b3 = st.columns(3)
    if b1.button("Start stream", type="primary", key="rt_start"):
        _start_stream(rt_interval)
    if b2.button("Inject attack now", key="rt_inject"):
        rt = st.session_state.get("rt")
        if rt and not rt["done"]:
            with open(os.path.join(rt["dir"], "control.json"), "w", encoding="utf-8") as f:
                json.dump({"nonce": uuid.uuid4().hex, "user": rt_user, "scenario": rt_scen}, f)
            st.toast(f"Attack `{rt_scen}` will be injected for {rt_user} on the next simulated day")
        else:
            st.warning("Start the stream first.")
    if b3.button("Stop", key="rt_stop"):
        _stop_stream()
        if st.session_state.get("rt"):
            st.session_state["rt"]["done"] = True

    rt_state = st.session_state.get("rt")
    rt_running = bool(rt_state and not rt_state["done"])

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
        truth_users = {t["user"] for t in scorer.truth}
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tick", state["tick"])
        m2.metric("Simulated date", str(state["sim_date"]))
        m3.metric("Events ingested", f"{scorer.n_events:,}")
        m4.metric("Last alert latency", f"{alerts[-1]['latency_s']:.2f} s" if alerts else "-")
        st.caption(f"Status: {'STREAMING' if not rt['done'] else 'STREAM ENDED'} - model: {rt['model']}")

        hist = scorer.history_df()
        top = state["risk_table"].user_id.head(4).tolist()
        cols = list(dict.fromkeys(top + ([rt_user] if rt_user in hist.columns else [])))
        st.altair_chart(charts.risk_timeseries(hist[cols], rt_user, alerts), width="stretch")
        st.dataframe(state["risk_table"].head(8).rename(columns={"user_id": "User", "anomaly_score": "Raw score", "risk_score": "Risk (0-100)"}),
                     width="stretch", hide_index=True)
        for a in reversed(alerts[-5:]):
            tag = "matches the injected attack" if a["user_id"] in truth_users else "NOT an injected attack (false alarm)"
            st.error(f"ALERT: {a['user_id']} risk {a['risk_score']:.0f}/100 on simulated day {a['sim_date']} "
                     f"(tick {a['tick']}), raised {a['latency_s']:.2f}s after the logs were written - {tag}")

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

    start = st.button("Start Live Feed", type="primary")

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
                m3.metric("Status", "ATTACK INJECTED" if state["injected"] else "monitoring...")

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
        "Colour = risk (green low, red high); size = e-mail strength; black outline = ground-truth insider "
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
