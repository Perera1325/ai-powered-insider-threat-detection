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

import sys
import time

import streamlit as st

sys.path.insert(0, "src")
import dashboard_utils as du
import live_demo as ld
from inject_anomalies import SCENARIOS

st.set_page_config(page_title="Insider Threat Detection - SOC Dashboard", layout="wide")


@st.cache_data
def _load_data():
    return du.load_data()


@st.cache_data
def _feature_matrix(_features):
    return du.get_feature_matrix(_features)


@st.cache_resource
def _trained_models(_X):
    return du.train_models(_X)


data = _load_data()
X, feature_cols, train_mean, train_std = _feature_matrix(data["features"])
models = _trained_models(X)
user_ids = data["features"].user_id.values

st.title("AI-Powered Insider Threat Detection")
st.caption(
    "Simulated organizational environment - all users, logs, and incidents are "
    "synthetically generated for this final year project. No real data is used."
)

with st.sidebar:
    st.header("Controls")
    model_name = st.selectbox("Detection model", du.MODEL_CHOICES, index=0)
    top_k = st.slider("Number of alerts to show", min_value=5, max_value=30, value=15)
    st.divider()
    st.caption(
        "`is_malicious` / `scenario` columns show the ground-truth label "
        "injected during dataset generation (Day 1) - shown here only "
        "because this is a research/demo dataset with known answers. A "
        "real deployment would not have this column; it exists to let "
        "you validate the model against a known answer key."
    )

tab_alerts, tab_live, tab_graph, tab_models = st.tabs(
    ["🚨 Alerts", "🔴 Live Demo", "🕸️ Interaction Graph", "📊 Model Comparison"]
)

with tab_alerts:
    alert_df = du.build_alert_table(data["features"], data["labels"], models, model_name, X, top_k=top_k)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Users monitored", len(data["features"]))
    col2.metric("Alerts shown", len(alert_df))
    col3.metric("High risk", int((alert_df.risk_level == "High").sum()))
    col4.metric("Flagged = ground truth match", int(alert_df.is_malicious.sum()))

    st.subheader(f"Top {top_k} alerts - {model_name}")
    st.dataframe(
        alert_df[["rank", "user_id", "risk_score", "risk_level", "is_malicious", "scenario"]],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Explain an alert")
    selected_uid = st.selectbox("Select a user to explain", alert_df.user_id.tolist())

    if selected_uid:
        row = alert_df[alert_df.user_id == selected_uid].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk score", f"{row.risk_score:.1f}/100")
        c2.metric("Risk level", row.risk_level)
        c3.metric("Ground-truth label", "Malicious" if row.is_malicious else "Normal")

        with st.spinner("Computing SHAP / LIME explanation..."):
            shap_s, lime_s = du.explain_user(models, model_name, X, feature_cols, user_ids, selected_uid)
            fig = du.explanation_figure(shap_s, lime_s, selected_uid)
        st.pyplot(fig)
        st.caption(
            "SHAP (left) shows each feature's contribution to this user's anomaly "
            "score relative to the population average. LIME (right) shows a local "
            "linear approximation around this specific user - see docs/day6_progress.md "
            "for why LIME is noisier than SHAP on this model."
        )

        st.subheader(f"{selected_uid}'s resource/relationship neighbourhood")
        G = du.build_full_graph(data)
        subfig = du.subgraph_figure(G, selected_uid, radius=1)
        if subfig:
            st.pyplot(subfig)

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

    start = st.button("▶ Start Live Feed", type="primary")

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
                m3.metric("Status", "🔴 ATTACK INJECTED" if state["injected"] else "monitoring...")

            fig = ld.plot_risk_history(history, demo_user_id, inject_at_tick)
            chart_placeholder.pyplot(fig)

            table_placeholder.dataframe(
                state["risk_table"].head(8).rename(
                    columns={"user_id": "User", "anomaly_score": "Raw score", "risk_score": "Risk (0-100)"}
                ),
                use_container_width=True,
                hide_index=True,
            )

            if state["injected"] or (state["tick"] > inject_at_tick and demo_row.risk_score >= 70):
                alert_placeholder.error(
                    f"🚨 ALERT: {demo_user_id} flagged HIGH RISK on simulated day {state['tick']} "
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
    st.subheader("Full user-resource interaction graph")
    st.image(
        "docs/figures/user_resource_graph.png",
        caption="Red = known malicious users, blue = normal users, yellow squares = shared resources",
        use_container_width=True,
    )
    st.caption(
        "See docs/day5_progress.md - graph centrality alone does not separate "
        "malicious users in the current dataset; documented there as an honest "
        "limitation with a concrete fix recommendation."
    )

with tab_models:
    st.subheader("Model comparison (Day 4 + Day 5)")
    st.dataframe(data["model_comparison"].round(3), use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.image("docs/figures/day4_roc_curves.png", caption="Day 4 ROC curves", use_container_width=True)
    with col2:
        st.image("docs/figures/day5_autoencoder.png", caption="Day 5 Autoencoder training + ROC", use_container_width=True)
    st.caption(
        "All models achieve perfect separation on this dataset - read "
        "docs/day4_progress.md's caveat before quoting this as real-world accuracy."
    )
