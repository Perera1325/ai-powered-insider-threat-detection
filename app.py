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

import streamlit as st

sys.path.insert(0, "src")
import dashboard_utils as du

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
X, feature_cols = _feature_matrix(data["features"])
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

tab_alerts, tab_graph, tab_models = st.tabs(["🚨 Alerts", "🕸️ Interaction Graph", "📊 Model Comparison"])

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
