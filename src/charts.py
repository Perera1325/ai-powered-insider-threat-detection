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


def live_leaderboard(table, threshold=THRESHOLD, n=10):
    """Horizontal bars of the current top-n risk scores; bars at/above the threshold are dark navy."""
    t = table.head(n).copy()
    t["state"] = ["Alert" if r >= threshold else "Normal" for r in t.risk_score]
    bars = (alt.Chart(t).mark_bar(size=16)
            .encode(y=alt.Y("user_id:N", sort=t.user_id.tolist(), title=None, axis=alt.Axis(labelOverlap=False)),
                    x=alt.X("risk_score:Q", title="Current risk (0-100)", scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color("state:N", scale=alt.Scale(domain=["Normal", "Alert"], range=["#93C5FD", "#1E3A8A"]),
                                    legend=alt.Legend(title=None)),
                    tooltip=["user_id", "risk_score"]))
    labels = bars.mark_text(align="left", dx=4, color="#1B2433", fontSize=11).encode(text=alt.Text("risk_score:Q", format=".0f"), color=alt.value("#1B2433"))
    rule = alt.Chart(pd.DataFrame({"x": [threshold]})).mark_rule(color="#DC2626", strokeDash=[5, 4], size=1.4).encode(x="x:Q")
    return _base_config((bars + labels + rule).properties(height=alt.Step(26), title="Live risk leaderboard"))


def risk_heatmap(hist, n_users=12):
    """Users x time heat map of risk (rows ordered by latest risk)."""
    if hist.empty:
        return None
    hist = hist.select_dtypes("number")
    order = hist.iloc[-1].sort_values(ascending=False).index.tolist()[:n_users]
    long = hist[order].reset_index().melt(id_vars="tick", var_name="User", value_name="Risk")
    ch = (alt.Chart(long).mark_rect()
          .encode(x=alt.X("tick:O", title="Tick (simulated working day)", axis=alt.Axis(labelOverlap=True)),
                  y=alt.Y("User:N", sort=order, title=None, axis=alt.Axis(labelOverlap=False)),
                  color=alt.Color("Risk:Q", scale=alt.Scale(domain=[0, 50, 100], range=["#EFF6FF", "#60A5FA", "#1E3A8A"]),
                                  legend=alt.Legend(title="Risk", gradientLength=110)),
                  tooltip=["User", "tick", alt.Tooltip("Risk:Q", format=".1f")]))
    return _base_config(ch.properties(height=alt.Step(22), title="Risk heat map (users x time)"))


def latency_chart(stats):
    """Per-tick alert-pipeline latency (producer write -> scored) with the 5 s target marked."""
    df = pd.DataFrame([s for s in stats if s["tick"] > 0])
    if df.empty:
        return None
    line = (alt.Chart(df).mark_line(point=True, color="#2563EB")
            .encode(x=alt.X("tick:Q", title="Tick"), y=alt.Y("latency_s:Q", title="Latency (s)", scale=alt.Scale(domain=[0, 5.5])),
                    tooltip=["tick", "latency_s", "score_ms", "events"]))
    rule = alt.Chart(pd.DataFrame({"y": [5]})).mark_rule(color="#DC2626", strokeDash=[5, 4]).encode(y="y:Q")
    return _base_config((line + rule).properties(height=170, title="Pipeline latency per tick (target < 5 s)"))


BLUE, NAVY_C, RED_C = "#2563EB", "#1E3A8A", "#DC2626"


def threshold_chart(sweep, current):
    """Precision / recall / F1 against the alert threshold, with the chosen threshold marked."""
    long = sweep.melt(id_vars="threshold", value_vars=["precision", "recall", "f1"], var_name="metric", value_name="value")
    lines = (alt.Chart(long).mark_line(strokeWidth=2.2, interpolate="monotone")
             .encode(x=alt.X("threshold:Q", title="Alert threshold (risk 0-100)", scale=alt.Scale(domain=[0, 100])),
                     y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[0, 1.02])),
                     color=alt.Color("metric:N", scale=alt.Scale(domain=["precision", "recall", "f1"],
                                                                 range=["#93C5FD", BLUE, NAVY_C]), legend=alt.Legend(title=None)),
                     tooltip=["metric", "threshold", alt.Tooltip("value:Q", format=".2f")]))
    rule = alt.Chart(pd.DataFrame({"x": [current]})).mark_rule(color=RED_C, strokeDash=[5, 4], size=1.4).encode(x="x:Q")
    return _base_config((lines + rule).properties(height=260, title="Precision / recall vs alert threshold"))


def alert_volume_chart(sweep, current):
    """Stacked true / false positives per threshold: the analyst workload view."""
    long = sweep.melt(id_vars="threshold", value_vars=["true_positives", "false_positives"], var_name="kind", value_name="users")
    long["kind"] = long.kind.map({"true_positives": "Real insiders flagged", "false_positives": "False alarms"})
    area = (alt.Chart(long).mark_area(opacity=.9, interpolate="step-after")
            .encode(x=alt.X("threshold:Q", title="Alert threshold (risk 0-100)", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("users:Q", title="Users alerted", stack=True),
                    color=alt.Color("kind:N", scale=alt.Scale(domain=["Real insiders flagged", "False alarms"],
                                                              range=[NAVY_C, "#BFDBFE"]), legend=alt.Legend(title=None)),
                    tooltip=["kind", "threshold", "users"]))
    rule = alt.Chart(pd.DataFrame({"x": [current]})).mark_rule(color=RED_C, strokeDash=[5, 4], size=1.4).encode(x="x:Q")
    return _base_config((area + rule).properties(height=260, title="Analyst workload at each threshold"))


def agreement_heatmap(ov):
    long = ov.reset_index(names="a").melt(id_vars="a", var_name="b", value_name="overlap")
    base = alt.Chart(long).encode(x=alt.X("b:N", title=None, axis=alt.Axis(labelAngle=-40, labelLimit=200, labelOverlap=False)),
                                  y=alt.Y("a:N", title=None, axis=alt.Axis(labelLimit=200, labelOverlap=False)))
    rect = base.mark_rect().encode(color=alt.Color("overlap:Q", scale=alt.Scale(domain=[0, 1], range=["#EFF6FF", "#60A5FA", NAVY_C]),
                                                    legend=None), tooltip=["a", "b", alt.Tooltip("overlap:Q", format=".2f")])
    txt = base.mark_text(fontSize=12).encode(text=alt.Text("overlap:Q", format=".2f"),
                                              color=alt.condition(alt.datum.overlap > .55, alt.value("white"), alt.value("#1B2433")))
    return _base_config((rect + txt).properties(height=alt.Step(36), title="Top-k overlap between models (Jaccard)"))


def deviation_chart(dev):
    d = dev.copy()
    d["direction"] = ["Above population" if z >= 0 else "Below population" for z in d.z_score]
    return _base_config(
        alt.Chart(d).mark_bar(size=15)
        .encode(y=alt.Y("feature:N", sort=d.feature.tolist(), title=None, axis=alt.Axis(labelLimit=240, labelOverlap=False)),
                x=alt.X("z_score:Q", title="Standard deviations from the population mean"),
                color=alt.Color("direction:N", scale=alt.Scale(domain=["Above population", "Below population"], range=[NAVY_C, "#93C5FD"]),
                                legend=alt.Legend(title=None)),
                tooltip=["feature", alt.Tooltip("value:Q", format=".2f"), alt.Tooltip("population_mean:Q", format=".2f"),
                         alt.Tooltip("z_score:Q", format=".1f")])
        .properties(height=alt.Step(24), title="Most unusual behaviours (z-score)"))


def timeline_chart(long, injected_day=None):
    """injected_day: ground-truth attack date (demo data only) drawn as a red rule."""
    if long.empty:
        return None
    pal = alt.Scale(domain=["File events", "E-mails sent", "External e-mails"], range=["#93C5FD", BLUE, NAVY_C])
    ch = (alt.Chart(long).mark_line(strokeWidth=1.8, interpolate="monotone")
          .encode(x=alt.X("date:T", title=None), y=alt.Y("count:Q", title="Events per day"),
                  color=alt.Color("series:N", scale=pal, legend=alt.Legend(title=None)),
                  tooltip=["series", alt.Tooltip("date:T", format="%d %b"), "count"]))
    if injected_day is not None and pd.notna(injected_day):
        ds = pd.DataFrame({"date": [pd.Timestamp(injected_day)]})
        ch = ch + alt.Chart(ds).mark_rule(color=RED_C, strokeDash=[4, 4]).encode(x="date:T")
    return _base_config(ch.properties(height=230, title="Daily activity timeline"))


def department_chart(dept):
    """Stacked-free bar chart: high-risk users and mean risk per department."""
    bars = (alt.Chart(dept).mark_bar(color="#2563EB", opacity=0.9)
            .encode(y=alt.Y("department:N", sort=dept.department.tolist(), title=None, axis=alt.Axis(labelOverlap=False)),
                    x=alt.X("mean_risk:Q", title="Mean risk (0-100)", scale=alt.Scale(domain=[0, 100])),
                    tooltip=["department", "users", "mean_risk", "max_risk", "high_risk_users"]))
    label = bars.mark_text(align="left", dx=4, color="#1B2433", fontSize=11).encode(
        text=alt.Text("high_risk_users:Q", format="d"), color=alt.value("#1B2433"))
    return _base_config((bars + label).properties(height=alt.Step(30), title="Mean risk by department (number = users above the threshold)"))


def case_funnel_chart(funnel):
    bars = (alt.Chart(funnel).mark_bar(color="#0891B2")
            .encode(x=alt.X("status:N", sort=funnel.status.tolist(), title=None, axis=alt.Axis(labelAngle=-25, labelLimit=170, labelOverlap=False)),
                    y=alt.Y("cases:Q", title="Cases", axis=alt.Axis(tickMinStep=1, format="d"),
                            scale=alt.Scale(domain=[0, max(2, int(funnel.cases.max()))])), tooltip=["status", "cases"]))
    return _base_config(bars.properties(height=220, title="Cases by status"))


def psi_chart(drift, n=12):
    t = drift.head(n).copy()
    bars = (alt.Chart(t).mark_bar(size=14)
            .encode(y=alt.Y("feature:N", sort=t.feature.tolist(), title=None, axis=alt.Axis(labelOverlap=False)),
                    x=alt.X("psi:Q", title="Population Stability Index"),
                    color=alt.Color("status:N", scale=alt.Scale(domain=["Within noise", "Stable", "Moderate", "Significant"],
                                                                range=["#CBD5E1", "#93C5FD", "#F59E0B", "#DC2626"]),
                                    legend=alt.Legend(title=None)),
                    tooltip=["feature", "psi", "status", "baseline_mean", "new_mean"]))
    rules = alt.Chart(pd.DataFrame({"x": [0.10, 0.25]})).mark_rule(color="#64748B", strokeDash=[4, 4]).encode(x="x:Q")
    return _base_config((bars + rules).properties(height=alt.Step(22), title="Feature drift against the baseline"))


def consensus_chart(cons, n=12):
    t = cons.head(n).copy()
    bars = (alt.Chart(t).mark_bar(size=14)
            .encode(y=alt.Y("user_id:N", sort=t.user_id.tolist(), title=None, axis=alt.Axis(labelOverlap=False)),
                    x=alt.X("consensus:Q", title="Consensus risk (mean of models)", scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color("confidence:N", scale=alt.Scale(domain=["High", "Medium", "Low", "-"],
                                                                    range=["#1E3A8A", "#3B82F6", "#93C5FD", "#CBD5E1"]),
                                    legend=alt.Legend(title="Model agreement")),
                    tooltip=["user_id", "consensus", "spread", "models_alerting", "confidence"]))
    return _base_config(bars.properties(height=alt.Step(22), title="Consensus risk across models"))
