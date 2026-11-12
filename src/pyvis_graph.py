"""
Interactive PyVis version of the user-resource graph (proposal: "Graph
Analysis - NetworkX, PyVis"). Produces a self-contained HTML file that
can be opened in any browser or embedded in the Streamlit dashboard.

Node colour = risk score (green -> red); square nodes = shared resources;
node size = degree. Hover a node to see its department, risk score and
the injected scenario (ground truth, demo dataset only).

Run:  python3 src/pyvis_graph.py   -> docs/figures/user_resource_graph.html
"""

import sys

import networkx as nx
import pandas as pd

sys.path.insert(0, "src")

DATA_DIR = "data/processed/full"
OUT_HTML = "docs/figures/user_resource_graph.html"


def _risk_colour(risk):
    """0 -> green, 100 -> red (hex)."""
    r = int(min(255, 2 * 255 * risk / 100))
    g = int(min(255, 2 * 255 * (1 - risk / 100)))
    return f"#{r:02x}{g:02x}50"


def build_pyvis_html(G, risk_by_user=None, labels=None, focus_user=None, height="650px", min_edge_weight=7, label_risk=50, top_contacts=12):
    """Returns the HTML string of an interactive PyVis network.

    risk_by_user : dict user_id -> 0..100 risk score (optional)
    labels       : DataFrame with user_id,is_malicious,scenario (optional)
    focus_user   : if given, draws a star: this user, their shared folders and their top_contacts strongest e-mail contacts
    min_edge_weight : hide user-user email links with fewer interactions than this
                      (default 7 keeps ~5% of links - the whole-org graph has ~4,700)
    label_risk   : in the whole-organisation view only users at/above this risk (and
                   ground-truth insiders) get a text label; everyone is still hoverable
    """
    from pyvis.network import Network

    risk_by_user = risk_by_user or {}
    label_map = {}
    if labels is not None:
        label_map = labels.set_index("user_id")[["is_malicious", "scenario"]].to_dict("index")

    # Focus view = a STAR: the user in the centre, their shared folders and their strongest
    # `top_contacts` e-mail contacts, with only the edges that touch the user. (2 hops is useless -
    # every user touches the same 4 folders so it reaches everyone - and drawing the links
    # between the contacts too gives ~1,000 edges.)
    if focus_user and focus_user in G:
        nbrs = sorted(G[focus_user].items(), key=lambda kv: kv[1].get("weight", 1), reverse=True)
        resources = [n for n, _ in nbrs if G.nodes[n].get("node_type") == "resource"]
        people = [n for n, _ in nbrs if G.nodes[n].get("node_type") != "resource"][:top_contacts]
        H = nx.Graph()
        for n in [focus_user, *resources, *people]:
            H.add_node(n, **G.nodes[n])
        for n in [*resources, *people]:
            H.add_edge(focus_user, n, **G[focus_user][n])
    else:
        H = G.copy()
        weak = [(u, v) for u, v, d in H.edges(data=True)
                if d.get("edge_type") == "email" and d.get("weight", 1) < min_edge_weight]
        H.remove_edges_from(weak)

    net = Network(height=height, width="100%", bgcolor="#ffffff", font_color="#222222", cdn_resources="in_line")
    for n, d in H.nodes(data=True):
        if d.get("node_type") == "resource":
            net.add_node(n, label=n.replace("folder:", ""), shape="box", color="#f2b134", mass=8,
                         font={"size": 18, "color": "#111111"}, title=f"Shared resource: {n}")
            continue
        risk = float(risk_by_user.get(n, 0.0))
        info = label_map.get(n, {})
        tip = f"{n} | dept: {d.get('department', '?')} | risk: {risk:.1f}/100"
        if info.get("is_malicious"):
            tip += f" | GROUND TRUTH: {info.get('scenario')}"
        size = 8 + 0.25 * H.degree(n)
        border = "#000000" if info.get("is_malicious") or n == focus_user else _risk_colour(risk)
        show_label = bool(focus_user) or risk >= label_risk or border == "#000000"
        net.add_node(n, label=n if show_label else " ", color={"background": _risk_colour(risk), "border": border},
                     borderWidth=3 if border == "#000000" else 1, size=size, title=tip,
                     font={"size": 16, "color": "#111111", "strokeWidth": 3, "strokeColor": "#ffffff"})
    for u, v, d in H.edges(data=True):
        is_file = d.get("edge_type") == "file_access"
        net.add_edge(u, v, width=0.4 if is_file else min(1 + float(d.get("weight", 1)) / 6, 4),
                     color="#e3e3e3" if is_file else "#9aa5b1",
                     title=f"{d.get('edge_type', '')} (weight {d.get('weight', 1)})")
    net.barnes_hut(gravity=-9000, central_gravity=0.15, spring_length=230, spring_strength=0.02,
                   damping=0.5, overlap=0.6)
    net.set_options('''{"physics": {"stabilization": {"enabled": true, "iterations": 250, "fit": true}},
                       "edges": {"smooth": false}, "interaction": {"hover": true, "tooltipDelay": 80}}''')
    return net.generate_html(notebook=False)


def main():
    import dashboard_utils as du

    data = du.load_data()
    X, feature_cols, _, _ = du.get_feature_matrix(data["features"])
    models = du.train_models(X)
    _, risk = du.score_users(models, "Isolation Forest", X)
    risk_by_user = dict(zip(data["features"].user_id, risk))
    G = du.build_full_graph(data)
    html = build_pyvis_html(G, risk_by_user, data["labels"])
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {OUT_HTML} ({len(html)//1024} KB, {G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")


if __name__ == "__main__":
    main()
