"""MITRE ATT&CK mapping for the insider-threat scenarios and behavioural features.

The mapping is analyst guidance, not proof: a feature deviation *suggests* a technique to investigate.
Technique ids/names follow MITRE ATT&CK Enterprise (https://attack.mitre.org/). Pure functions, no Streamlit.
"""
import pandas as pd

TACTICS = ["Initial Access", "Collection", "Discovery", "Exfiltration"]

# id -> (name, tactic)
TECHNIQUES = {
    "T1078": ("Valid Accounts", "Initial Access"),
    "T1039": ("Data from Network Shared Drive", "Collection"),
    "T1005": ("Data from Local System", "Collection"),
    "T1083": ("File and Directory Discovery", "Discovery"),
    "T1069": ("Permission Groups Discovery", "Discovery"),
    "T1052.001": ("Exfiltration over USB", "Exfiltration"),
    "T1048": ("Exfiltration Over Alternative Protocol", "Exfiltration"),
}

# the five injected scenarios (src/inject_anomalies.py) -> techniques
SCENARIO_MAP = {
    "after_hours_login": ["T1078"],
    "mass_file_access": ["T1039", "T1005"],
    "usb_exfiltration": ["T1052.001"],
    "data_exfil_email": ["T1048"],
    "privilege_probe": ["T1069", "T1083"],
}

# behavioural features (28 per-user features) -> techniques they hint at
FEATURE_MAP = {
    "after_hours_logon_count": ["T1078"], "after_hours_logon_ratio": ["T1078"],
    "avg_logon_hour": ["T1078"], "std_logon_hour": ["T1078"],
    "device_connect_count": ["T1052.001"], "device_after_hours_count": ["T1052.001"],
    "external_email_count": ["T1048"], "external_email_ratio": ["T1048"],
    "max_external_size_kb": ["T1048"], "total_attachments": ["T1048"],
    "max_daily_shared_folder_events": ["T1039"], "shared_folder_categories_touched": ["T1039", "T1069"],
    "max_daily_file_events": ["T1005"], "copy_count": ["T1005"], "copy_ratio": ["T1005"],
    "file_event_count": ["T1005"], "distinct_files": ["T1083"],
}


def technique_url(tid):
    return "https://attack.mitre.org/techniques/" + tid.replace(".", "/") + "/"


def technique_label(tid):
    name = TECHNIQUES.get(tid, ("", ""))[0]
    return f"{tid} {name}".strip()


def for_scenario(scenario):
    return list(SCENARIO_MAP.get(scenario, []))


def for_drivers(deviations, min_abs_z=2.0):
    """Rank techniques from a feature-deviation frame (columns feature, z_score).
    Only features with |z| >= min_abs_z vote; score = sum of |z| of matching features."""
    votes = {}
    for r in deviations.itertuples(index=False):
        if abs(r.z_score) < min_abs_z:
            continue
        for tid in FEATURE_MAP.get(r.feature, []):
            v = votes.setdefault(tid, {"score": 0.0, "features": []})
            v["score"] += abs(float(r.z_score))
            v["features"].append(r.feature)
    rows = [{"technique": t, "name": TECHNIQUES[t][0], "tactic": TECHNIQUES[t][1],
             "evidence_score": round(v["score"], 1), "features": ", ".join(v["features"])}
            for t, v in votes.items()]
    cols = ["technique", "name", "tactic", "evidence_score", "features"]
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("evidence_score", ascending=False).reset_index(drop=True)


def tag_users(features, user_ids, feature_cols, deviations_fn, top_n=1):
    """{user_id: 'T1048 Exfiltration ...'} using the strongest technique(s) per user."""
    tags = {}
    for uid in user_ids:
        m = for_drivers(deviations_fn(features, uid, feature_cols, 10))
        tags[uid] = ", ".join(technique_label(t) for t in m.technique.head(top_n)) if len(m) else "-"
    return tags


def coverage_matrix(user_techniques):
    """user_techniques: {user: [technique ids]} -> DataFrame technique x (tactic, users flagged)."""
    rows = []
    for tid, (name, tactic) in TECHNIQUES.items():
        users = sorted(u for u, ts in user_techniques.items() if tid in ts)
        rows.append({"tactic": tactic, "technique": tid, "name": name, "users_flagged": len(users),
                     "users": ", ".join(users[:8]) + (" ..." if len(users) > 8 else "")})
    df = pd.DataFrame(rows)
    df["tactic"] = pd.Categorical(df["tactic"], TACTICS, ordered=True)
    return df.sort_values(["tactic", "technique"]).reset_index(drop=True)


def from_driver_text(text):
    """Technique ids suggested by a '<feature> (+z sd), ...' string (used for real-time alerts)."""
    text = text or ""
    hits = sorted((text.index(f), f) for f in FEATURE_MAP if f in text)  # strongest driver (listed first) wins
    found = []
    for _, feat in hits:
        for t in FEATURE_MAP[feat]:
            if t not in found:
                found.append(t)
    return found


def user_matches(features, user_ids, feature_cols, deviations_fn):
    """{user_id: DataFrame from for_drivers} for each user (features = the full population frame)."""
    return {u: for_drivers(deviations_fn(features, u, feature_cols, 10)) for u in user_ids}


def agreement(matches, labels):
    """Validation against the synthetic answer key: of the known insiders present in `matches`, how many have
    a suggested technique that belongs to their injected scenario. labels: frame with user_id, scenario, is_malicious."""
    truth = labels[labels.is_malicious.astype(bool)].set_index("user_id")["scenario"].to_dict()
    pairs = [(u, truth[u]) for u in matches if u in truth]
    hit = sum(1 for u, s in pairs if set(matches[u].technique) & set(for_scenario(s)))
    return hit, len(pairs)
