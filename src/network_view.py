"""Scientific network view of the user-resource graph (Altair, deterministic layout).

Replaces the physics-based PyVis view in the dashboard: no jitter, no clipping, same result every
run. Departments are placed as clusters on a ring, shared folders sit in the centre, node colour =
risk, node size = e-mail strength (sum of link weights), black outline = ground-truth insider (demo
data only). Graph statistics (density, modularity, centralities, risk-vs-centrality correlation)
are computed with NetworkX so the picture can be backed by numbers.
"""
import math

import altair as alt
import networkx as nx
import numpy as np
import pandas as pd

RISK_SCALE = alt.Scale(domain=[0, 40, 70, 100], range=["#DBEAFE", "#93C5FD", "#2563EB", "#1E3A8A"], clamp=True)
EDGE_KINDS = ["Within department", "Cross-department", "Involves flagged user", "File access (flagged user)"]
EDGE_COLOURS = ["#94A3B8", "#475569", "#1D4ED8", "#60A5FA"]


def _users(G):
    return [n for n, d in G.nodes(data=True) if d.get("node_type") != "resource"]


def _email_graph(G, min_w=1):
    H = nx.Graph()
    H.add_nodes_from((n, G.nodes[n]) for n in _users(G))
    for u, v, d in G.edges(data=True):
        if d.get("edge_type") == "email" and d.get("weight", 1) >= min_w:
            H.add_edge(u, v, weight=d.get("weight", 1))
    return H


def layout(G):
    """Deterministic cluster layout: department clusters on a ring, folders in the centre."""
    depts = sorted({G.nodes[n].get("department", "?") for n in _users(G)})
    pos, centres = {}, {}
    for k, dpt in enumerate(depts):
        ang = 2 * math.pi * k / len(depts) - math.pi / 2
        cx, cy = 3.6 * math.cos(ang), 3.6 * math.sin(ang)
        centres[dpt] = (cx, cy, ang)
        members = sorted(n for n in _users(G) if G.nodes[n].get("department") == dpt)
        rad = 0.35 + 0.045 * len(members)
        for i, n in enumerate(members):
            a = 2 * math.pi * i / len(members)
            pos[n] = (cx + rad * math.cos(a), cy + rad * math.sin(a))
    res = sorted(n for n, d in G.nodes(data=True) if d.get("node_type") == "resource")
    for i, n in enumerate(res):
        a = 2 * math.pi * i / max(len(res), 1) + math.pi / 4
        pos[n] = (0.75 * math.cos(a), 0.75 * math.sin(a))
    return pos, centres


def node_table(G, risk_by_user, labels=None, min_w=7):
    """Per-user table: department, role-free metrics (strength, betweenness, clustering), risk."""
    Hall = _email_graph(G, 1)
    Hthr = _email_graph(G, min_w)
    strength = dict(Hall.degree(weight="weight"))
    btw = nx.betweenness_centrality(Hthr)
    clu = nx.clustering(Hthr)
    info = {}
    if labels is not None:
        info = labels.set_index("user_id")[["is_malicious", "scenario"]].to_dict("index")
    rows = []
    for n in _users(G):
        rows.append({"user": n, "department": G.nodes[n].get("department", "?"),
                     "risk": float(risk_by_user.get(n, 0.0)), "strength": int(strength.get(n, 0)),
                     "links_shown": int(Hthr.degree(n)), "betweenness": round(float(btw.get(n, 0.0)), 4),
                     "clustering": round(float(clu.get(n, 0.0)), 3),
                     "insider": bool(info.get(n, {}).get("is_malicious", False)),
                     "scenario": ("" if pd.isna(info.get(n, {}).get("scenario")) else str(info[n]["scenario"]))})
    return pd.DataFrame(rows)


def graph_stats(G, table, min_w=7):
    H = _email_graph(G, min_w)
    groups = {}
    for n, d in H.nodes(data=True):
        groups.setdefault(d.get("department", "?"), set()).add(n)
    try:
        mod = nx.community.modularity(H, list(groups.values()), weight="weight") if H.number_of_edges() else float("nan")
    except Exception:
        mod = float("nan")
    rho = table["risk"].corr(table["strength"], method="spearman")
    return {"nodes": H.number_of_nodes(), "edges": H.number_of_edges(), "density": nx.density(H),
            "components": nx.number_connected_components(H), "modularity": mod,
            "spearman_risk_strength": rho}


def network_chart(G, table, min_w=7, label_risk=50, focus=None, height=620):
    pos, centres = layout(G)
    t = table.set_index("user")
    people = [n for n in _users(G)]
    top_flag = set(t.index[t["risk"] >= label_risk])
    nbrs = set()
    edges = []
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1)
        if d.get("edge_type") == "email":
            if focus:
                if focus not in (u, v):
                    continue
            elif w < min_w:
                continue
            same = G.nodes[u].get("department") == G.nodes[v].get("department")
            kind = "Involves flagged user" if (u in top_flag or v in top_flag) else ("Within department" if same else "Cross-department")
        else:
            person = u if G.nodes[u].get("node_type") != "resource" else v
            if focus:
                if person != focus:
                    continue
            elif person not in top_flag:
                continue
            kind = "File access (flagged user)"
        if focus:
            nbrs.update([u, v])
        edges.append({"x": pos[u][0], "y": pos[u][1], "x2": pos[v][0], "y2": pos[v][1], "kind": kind, "weight": float(w),
                      "link": f"{u} – {v}"})
    edf = pd.DataFrame(edges)
    ndf = pd.DataFrame([{"user": n, "x": pos[n][0], "y": pos[n][1], "department": t.loc[n, "department"],
                         "risk": round(t.loc[n, "risk"], 1), "strength": int(t.loc[n, "strength"]),
                         "betweenness": float(t.loc[n, "betweenness"]), "insider": bool(t.loc[n, "insider"]),
                         "scenario": t.loc[n, "scenario"] or "-",
                         "label": n if (n in top_flag or t.loc[n, "insider"] or n == focus) else "",
                         "dim": bool(focus and n not in nbrs and n != focus)} for n in people])
    rdf = pd.DataFrame([{"x": pos[n][0], "y": pos[n][1], "name": n.replace("folder:", "")} for n in pos if n.startswith("folder:")])
    ddf = pd.DataFrame([{"x": (cx + 1.55 * math.cos(a)), "y": (cy + 1.55 * math.sin(a)), "department": d}
                        for d, (cx, cy, a) in centres.items()])
    lim = [-5.6, 5.6]
    xs = alt.X("x:Q", scale=alt.Scale(domain=lim), axis=None)
    ys = alt.Y("y:Q", scale=alt.Scale(domain=lim), axis=None)
    layers = []
    if len(edf):
        layers.append(alt.Chart(edf).mark_rule().encode(
            x=xs, y=ys, x2="x2:Q", y2="y2:Q",
            color=alt.Color("kind:N", scale=alt.Scale(domain=EDGE_KINDS, range=EDGE_COLOURS),
                            legend=alt.Legend(title="Link type", symbolType="stroke")),
            size=alt.Size("weight:Q", scale=alt.Scale(range=[0.5, 2.6]), legend=None),
            opacity=alt.Opacity("kind:N", scale=alt.Scale(domain=EDGE_KINDS, range=[0.38, 0.32, 0.75, 0.7] if not focus else [0.7] * 4), legend=None),
            tooltip=["link:N", "kind:N", "weight:Q"]))
    layers.append(alt.Chart(ddf).mark_text(fontSize=12, fontWeight="bold", color="#334155").encode(x=xs, y=ys, text="department:N"))
    layers.append(alt.Chart(rdf).mark_square(size=900, color="#DBEAFE", opacity=1, stroke="#1E3A8A", strokeWidth=1.2).encode(
        x=xs, y=ys, tooltip=[alt.Tooltip("name:N", title="Shared folder")]))
    layers.append(alt.Chart(rdf).mark_text(fontSize=11, color="#1E3A8A", fontWeight="bold").encode(x=xs, y=ys, text="name:N"))
    layers.append(alt.Chart(ndf).mark_circle().encode(
        x=xs, y=ys,
        size=alt.Size("strength:Q", scale=alt.Scale(range=[60, 420]), legend=alt.Legend(title="E-mail strength")),
        color=alt.Color("risk:Q", scale=RISK_SCALE, legend=alt.Legend(title="Risk (0-100)", gradientLength=160)),
        stroke=alt.condition("datum.insider", alt.value("#000000"), alt.value("#FFFFFF")),
        strokeWidth=alt.condition("datum.insider", alt.value(3), alt.value(1)),
        opacity=alt.condition("datum.dim", alt.value(0.18), alt.value(0.95)),
        tooltip=[alt.Tooltip("user:N", title="User"), "department:N", alt.Tooltip("risk:Q", format=".1f"),
                 "strength:Q", alt.Tooltip("betweenness:Q", format=".3f"),
                 alt.Tooltip("insider:N", title="Ground-truth insider (demo)"), alt.Tooltip("scenario:N", title="Scenario")]))
    layers.append(alt.Chart(ndf[ndf.label != ""]).mark_text(dy=-13, fontSize=11, fontWeight="bold", color="#111827").encode(
        x=xs, y=ys, text="label:N"))
    ch = (alt.layer(*layers).resolve_scale(size="independent", color="independent", opacity="independent")
          .properties(height=height).interactive())
    return (ch.configure_view(strokeWidth=0, fill="#FFFFFF")
              .configure_legend(labelColor="#3A4A63", titleColor="#3A4A63", orient="right"))


def risk_vs_strength_chart(table, rho, height=330):
    base = alt.Chart(table).encode(
        x=alt.X("strength:Q", title="E-mail strength (sum of link weights)"),
        y=alt.Y("risk:Q", title="Risk score (0-100)", scale=alt.Scale(domain=[0, 105])))
    pts = base.mark_circle(size=90, opacity=0.85).encode(
        color=alt.Color("insider:N", title="Ground-truth insider", scale=alt.Scale(domain=[False, True], range=["#93C5FD", "#1E3A8A"])),
        tooltip=["user:N", "department:N", alt.Tooltip("risk:Q", format=".1f"), "strength:Q", "scenario:N"])
    fit = base.transform_regression("strength", "risk").mark_line(color="#64748B", strokeDash=[5, 4])
    thr = alt.Chart(pd.DataFrame({"y": [70]})).mark_rule(color="#DC2626", strokeDash=[4, 4]).encode(y="y:Q")
    return (pts + fit + thr).properties(height=height, title=f"Risk vs network strength (Spearman rho = {rho:.2f})").configure_axis(
        gridColor="#E5EAF2", labelColor="#5B6B82", titleColor="#3A4A63").configure_title(color="#1B2433", fontSize=13, anchor="start").configure_view(strokeWidth=0)
