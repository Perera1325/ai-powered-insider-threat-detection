"""
Day 5 (part 2) - Graph analysis with NetworkX.

Builds a bipartite user-resource interaction graph from the file-access
and email logs: users are connected to the shared resources they touch
(shared folder categories, PCs, and email correspondents). Computes
degree and betweenness centrality to see whether the known malicious
users occupy structurally unusual positions - e.g. touching resources
outside their normal circle (privilege_probe), or having disproportionately
high degree (mass_file_access, usb_exfiltration).

Outputs:
  - docs/figures/user_resource_graph.png (full graph, malicious users highlighted)
  - data/processed/full/graph_centrality.csv (per-user centrality features)
  - docs/day5_graph_summary.md
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

DATA_DIR = "data/processed/full"
FIG_DIR = "docs/figures"

SHARED_FOLDER_CATEGORIES = ["reports", "hr", "finance", "projects"]


def _folder_category(path):
    path_lower = str(path).lower()
    for cat in SHARED_FOLDER_CATEGORIES:
        if f"\\{cat}\\" in path_lower:
            return cat
    return None


def build_graph(file_df, email_df, users):
    G = nx.Graph()
    for _, u in users.iterrows():
        G.add_node(u.user_id, node_type="user", department=u.department)

    # user -> shared-folder-category edges (weighted by access count)
    f = file_df.copy()
    f["category"] = f.filename.apply(_folder_category)
    shared = f[f.category.notna()]
    folder_counts = shared.groupby(["user_id", "category"]).size().reset_index(name="weight")
    for cat in folder_counts.category.unique():
        G.add_node(f"folder:{cat}", node_type="resource")
    for _, row in folder_counts.iterrows():
        G.add_edge(row.user_id, f"folder:{row.category}", weight=row.weight, edge_type="file_access")

    # user -> user edges via internal email (sender -> each internal recipient)
    internal_email = email_df[email_df.external == False]  # noqa: E712
    for _, row in internal_email.iterrows():
        recipients = [r.split("@")[0].upper() for r in str(row.to).split(";") if "@" in r]
        for r in recipients:
            r_id = r.replace("U", "U") if r.startswith("U") else None
            if r_id and G.has_node(r_id) and r_id != row.user_id:
                if G.has_edge(row.user_id, r_id):
                    G[row.user_id][r_id]["weight"] += 1
                else:
                    G.add_edge(row.user_id, r_id, weight=1, edge_type="email")
    return G


def main():
    file_df = pd.read_csv(f"{DATA_DIR}/file.csv", parse_dates=["timestamp"])
    email_df = pd.read_csv(f"{DATA_DIR}/email.csv", parse_dates=["timestamp"])
    users = pd.read_csv(f"{DATA_DIR}/users.csv")
    labels = pd.read_csv(f"{DATA_DIR}/labels.csv")

    G = build_graph(file_df, email_df, users)

    user_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "user"]
    degree = dict(G.degree(user_nodes, weight=None))
    weighted_degree = dict(G.degree(user_nodes, weight="weight"))
    betweenness = nx.betweenness_centrality(G, k=min(60, G.number_of_nodes()), seed=42)

    # how many DISTINCT shared-folder categories has each user touched (resource diversity)
    resource_diversity = {
        u: sum(1 for n in G.neighbors(u) if str(n).startswith("folder:")) for u in user_nodes
    }

    centrality_df = pd.DataFrame(
        {
            "user_id": user_nodes,
            "graph_degree": [degree[u] for u in user_nodes],
            "graph_weighted_degree": [weighted_degree[u] for u in user_nodes],
            "graph_betweenness": [betweenness[u] for u in user_nodes],
            "resource_diversity": [resource_diversity[u] for u in user_nodes],
        }
    ).merge(labels[["user_id", "is_malicious", "scenario"]], on="user_id", how="left")
    centrality_df["is_malicious"] = centrality_df["is_malicious"].fillna(False)
    centrality_df.to_csv(f"{DATA_DIR}/graph_centrality.csv", index=False)

    # --- visualize ---
    plt.figure(figsize=(11, 11))
    pos = nx.spring_layout(G, seed=42, k=0.3)

    resource_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "resource"]
    malicious_ids = set(labels[labels.is_malicious].user_id)
    normal_user_nodes = [u for u in user_nodes if u not in malicious_ids]
    malicious_user_nodes = [u for u in user_nodes if u in malicious_ids]

    nx.draw_networkx_edges(G, pos, alpha=0.08)
    nx.draw_networkx_nodes(G, pos, nodelist=resource_nodes, node_color="#f2b134", node_shape="s", node_size=350, label="shared resource")
    nx.draw_networkx_nodes(G, pos, nodelist=normal_user_nodes, node_color="#4a90d9", node_size=60, label="normal user")
    nx.draw_networkx_nodes(G, pos, nodelist=malicious_user_nodes, node_color="#d94a4a", node_size=220, label="malicious user")
    nx.draw_networkx_labels(G, pos, labels={n: n.replace("folder:", "") for n in resource_nodes}, font_size=9)
    nx.draw_networkx_labels(G, pos, labels={u: u for u in malicious_user_nodes}, font_size=7)

    plt.title("User-resource interaction graph (email + shared-folder access)")
    plt.legend(scatterpoints=1, loc="lower left", fontsize=9)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/user_resource_graph.png", dpi=120)
    plt.close()

    # summary
    lines = ["# Day 5 Graph Analysis Summary\n"]
    lines.append(f"Graph: {G.number_of_nodes()} nodes ({len(user_nodes)} users, "
                 f"{len(resource_nodes)} shared resources), {G.number_of_edges()} edges.\n")
    lines.append("## Centrality features by group (mean)\n")
    lines.append("```")
    lines.append(
        centrality_df.groupby("is_malicious")[
            ["graph_degree", "graph_weighted_degree", "graph_betweenness", "resource_diversity"]
        ].mean().round(4).to_string()
    )
    lines.append("```\n")
    lines.append("## Malicious users' graph features\n")
    lines.append("```")
    lines.append(
        centrality_df[centrality_df.is_malicious][
            ["user_id", "scenario", "graph_degree", "graph_weighted_degree", "graph_betweenness", "resource_diversity"]
        ].to_string(index=False)
    )
    lines.append("```")

    with open("docs/day5_graph_summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
