"""Altair charts for the dashboard (clean, light, SOC-style)."""
import altair as alt
import pandas as pd

THRESHOLD = 70
_PALETTE = ["#2563EB", "#0891B2", "#7C3AED", "#64748B", "#059669", "#DB2777"]


def _base_config(chart):
    return (chart.configure_view(strokeWidth=0)
            .configure_axis(gridColor="#E5EAF2", domainColor="#C9D2E0", tickColor="#C9D2E0",
                            labelColor="#5B6B82", titleColor="#3A4A63", labelFontSize=11, titleFontSize=11)
            .configure_legend(labelColor="#3A4A63", titleColor="#3A4A63", orient="bottom", direction="horizontal")
            .configure_title(color="#1B2433", fontSize=13, anchor="start"))


def _threshold_layer(x_field):
    rule = alt.Chart(pd.DataFrame({"y": [THRESHOLD]})).mark_rule(color="#DC2626", strokeDash=[5, 4], size=1.4).encode(y="y:Q")
    label = (alt.Chart(pd.DataFrame({"y": [THRESHOLD], "t": ["Alert threshold (70)"]}))
             .mark_text(align="left", dx=4, dy=-6, color="#B91C1C", fontSize=11, fontWeight="bold")
             .encode(y="y:Q", text="t:N", x=alt.value(4)))
    return rule + label


def risk_timeseries(hist, highlight, alerts=None, title="Risk score by entity"):
    """hist: DataFrame index=tick, columns=user ids (risk 0-100). highlight: user to emphasise.
    alerts: list of dicts with user_id, tick, risk_score."""
    long = hist.reset_index().melt(id_vars="tick", var_name="Entity", value_name="Risk")
    long["emph"] = long["Entity"] == highlight
    order = [highlight] + [c for c in hist.columns if c != highlight] if highlight in hist.columns else list(hist.columns)
    color = alt.Color("Entity:N", sort=order, scale=alt.Scale(domain=order, range=_PALETTE[: len(order)]),
                      legend=alt.Legend(title=None))
    lines = (alt.Chart(long).mark_line(interpolate="monotone")
             .encode(x=alt.X("tick:Q", title="Simulated day (tick)", axis=alt.Axis(tickMinStep=1)),
                     y=alt.Y("Risk:Q", title="Risk score (0-100)", scale=alt.Scale(domain=[0, 105])),
                     color=color, size=alt.condition("datum.emph", alt.value(3.2), alt.value(1.4)),
                     opacity=alt.condition("datum.emph", alt.value(1), alt.value(0.65)),
                     tooltip=["Entity", "tick", alt.Tooltip("Risk:Q", format=".1f")]))
    layers = lines + _threshold_layer("tick")
    if alerts:
        pts = pd.DataFrame(alerts)[["user_id", "tick", "risk_score"]].rename(columns={"user_id": "Entity", "risk_score": "Risk"})
        marks = (alt.Chart(pts).mark_point(shape="triangle-up", size=150, filled=True, color="#DC2626", stroke="white", strokeWidth=1.2)
                 .encode(x="tick:Q", y="Risk:Q", tooltip=["Entity", "tick", alt.Tooltip("Risk:Q", format=".1f")]))
        layers = layers + marks
    return _base_config(layers.properties(height=300, title=title))


def live_risk_chart(history, user, inject_tick):
    df = pd.DataFrame(history).rename(columns={"demo_user_risk": "Risk"})
    line = (alt.Chart(df).mark_line(color="#2563EB", size=3, point=alt.OverlayMarkDef(filled=True, size=70, color="#2563EB"))
            .encode(x=alt.X("tick:Q", title="Simulated day", axis=alt.Axis(tickMinStep=1)),
                    y=alt.Y("Risk:Q", title="Risk score (0-100)", scale=alt.Scale(domain=[0, 105])),
                    tooltip=["tick", alt.Tooltip("Risk:Q", format=".1f")]))
    inj = alt.Chart(pd.DataFrame({"x": [inject_tick]})).mark_rule(color="#D97706", strokeDash=[6, 4], size=1.6).encode(x="x:Q")
    inj_lbl = (alt.Chart(pd.DataFrame({"x": [inject_tick], "t": ["Attack injected"]}))
               .mark_text(align="left", dx=5, dy=12, color="#92400E", fontSize=11, fontWeight="bold")
               .encode(x="x:Q", y=alt.value(4), text="t:N"))
    return _base_config((line + _threshold_layer("tick") + inj + inj_lbl).properties(height=280, title=f"{user} risk score during the live feed"))


def risk_distribution(res):
    """Histogram of risk scores for an uploaded population, with the alert threshold marked."""
    bars = (alt.Chart(res).mark_bar(color="#2563EB", opacity=0.85)
            .encode(x=alt.X("risk_score:Q", bin=alt.Bin(step=5), title="Risk score (0-100)"),
                    y=alt.Y("count()", title="Users"), tooltip=["count()"]))
    rule = alt.Chart(pd.DataFrame({"x": [THRESHOLD]})).mark_rule(color="#DC2626", strokeDash=[5, 4], size=1.4).encode(x="x:Q")
    return _base_config((bars + rule).properties(height=220, title="Distribution of risk scores"))
