"""
Day 3 - Feature engineering.

Aggregates the five raw event logs (logon, device, http, email, file)
into a single per-user behavioral feature table, covering the whole
30-day window. This is the table Day 4's unsupervised models
(Isolation Forest, One-Class SVM) will train on.

IMPORTANT: labels.csv is joined in ONLY for the human-readable preview
CSV (user_features_preview_with_labels.csv) and the ranking sanity
check below. The clean, label-free `user_features.csv` is what actually
gets used for model training in Day 4.
"""

import numpy as np
import pandas as pd

DATA_DIR = "data/processed/full"

SHARED_FOLDER_CATEGORIES = ["reports", "hr", "finance", "projects"]


def _folder_category(path):
    path_lower = str(path).lower()
    for cat in SHARED_FOLDER_CATEGORIES:
        if f"\\{cat}\\" in path_lower:
            return cat
    return "personal"


def build_logon_features(logon):
    logon_only = logon[logon.activity == "Logon"].copy()
    logon_only["hour"] = logon_only.timestamp.dt.hour
    logon_only["date"] = logon_only.timestamp.dt.date
    logon_only["after_hours"] = (logon_only.hour < 6) | (logon_only.hour > 20)

    g = logon_only.groupby("user_id")
    feats = pd.DataFrame(
        {
            "total_logons": g.size(),
            "avg_logon_hour": g["hour"].mean(),
            "std_logon_hour": g["hour"].std().fillna(0),
            "after_hours_logon_count": g["after_hours"].sum(),
        }
    )
    feats["after_hours_logon_ratio"] = feats.after_hours_logon_count / feats.total_logons

    # session duration: pair each Logon with the next Logoff on the same pc/user
    sessions = []
    for (user_id, pc), grp in logon.groupby(["user_id", "pc"]):
        grp = grp.sort_values("timestamp")
        pending_logon = None
        for _, row in grp.iterrows():
            if row.activity == "Logon":
                pending_logon = row.timestamp
            elif row.activity == "Logoff" and pending_logon is not None:
                dur_min = (row.timestamp - pending_logon).total_seconds() / 60
                if 0 < dur_min < 24 * 60:
                    sessions.append({"user_id": user_id, "duration_min": dur_min})
                pending_logon = None
    sess_df = pd.DataFrame(sessions)
    if not sess_df.empty:
        sess_g = sess_df.groupby("user_id")["duration_min"]
        feats = feats.join(sess_g.mean().rename("avg_session_min"))
        feats = feats.join(sess_g.std().rename("std_session_min"))
    return feats.fillna(0)


def build_device_features(device):
    device_only = device[device.activity == "Connect"].copy()
    if device_only.empty:
        return pd.DataFrame(columns=["device_connect_count", "device_after_hours_count"])
    device_only["hour"] = device_only.timestamp.dt.hour
    device_only["after_hours"] = (device_only.hour < 6) | (device_only.hour > 20)
    g = device_only.groupby("user_id")
    return pd.DataFrame(
        {
            "device_connect_count": g.size(),
            "device_after_hours_count": g["after_hours"].sum(),
        }
    ).fillna(0)


def build_http_features(http):
    g = http.groupby("user_id")
    return pd.DataFrame(
        {
            "http_event_count": g.size(),
            "distinct_domains": g["url"].nunique(),
        }
    ).fillna(0)


def build_email_features(email):
    g = email.groupby("user_id")
    ext = email[email.external == True]  # noqa: E712
    ext_g = ext.groupby("user_id")
    feats = pd.DataFrame(
        {
            "email_count": g.size(),
            "avg_email_size_kb": g["size_kb"].mean(),
            "total_attachments": g["attachments"].sum(),
        }
    )
    feats = feats.join(ext_g.size().rename("external_email_count"))
    feats = feats.join(ext_g["size_kb"].max().rename("max_external_size_kb"))
    feats["external_email_count"] = feats["external_email_count"].fillna(0)
    feats["max_external_size_kb"] = feats["max_external_size_kb"].fillna(0)
    feats["external_email_ratio"] = feats.external_email_count / feats.email_count
    return feats.fillna(0)


def build_file_features(file_df):
    f = file_df.copy()
    f["date"] = f.timestamp.dt.date
    f["category"] = f.filename.apply(_folder_category)

    g = f.groupby("user_id")
    feats = pd.DataFrame(
        {
            "file_event_count": g.size(),
            "distinct_files": g["filename"].nunique(),
            "copy_count": f[f.action == "Copy"].groupby("user_id").size(),
            "delete_count": f[f.action == "Delete"].groupby("user_id").size(),
        }
    ).fillna(0)
    feats["copy_ratio"] = feats.copy_count / feats.file_event_count

    # burst detection: max single-day file event count per user
    daily_counts = f.groupby(["user_id", "date"]).size().rename("daily_count").reset_index()
    max_daily = daily_counts.groupby("user_id")["daily_count"].max().rename("max_daily_file_events")
    feats = feats.join(max_daily)

    # shared-folder diversity + burst within a single shared-folder category-day
    shared = f[f.category != "personal"]
    feats = feats.join(
        shared.groupby("user_id")["category"].nunique().rename("shared_folder_categories_touched")
    )
    cat_daily = (
        shared.groupby(["user_id", "date", "category"]).size().rename("count").reset_index()
    )
    if not cat_daily.empty:
        max_cat_daily = cat_daily.groupby("user_id")["count"].max().rename("max_daily_shared_folder_events")
        feats = feats.join(max_cat_daily)
    else:
        feats["max_daily_shared_folder_events"] = 0

    return feats.fillna(0)


def main():
    logon = pd.read_csv(f"{DATA_DIR}/logon.csv", parse_dates=["timestamp"])
    device = pd.read_csv(f"{DATA_DIR}/device.csv", parse_dates=["timestamp"])
    http = pd.read_csv(f"{DATA_DIR}/http.csv", parse_dates=["timestamp"])
    email = pd.read_csv(f"{DATA_DIR}/email.csv", parse_dates=["timestamp"])
    file_df = pd.read_csv(f"{DATA_DIR}/file.csv", parse_dates=["timestamp"])
    users = pd.read_csv(f"{DATA_DIR}/users.csv")
    labels = pd.read_csv(f"{DATA_DIR}/labels.csv")

    logon_feats = build_logon_features(logon)
    device_feats = build_device_features(device)
    http_feats = build_http_features(http)
    email_feats = build_email_features(email)
    file_feats = build_file_features(file_df)

    features = (
        users[["user_id"]]
        .set_index("user_id")
        .join(logon_feats)
        .join(device_feats)
        .join(http_feats)
        .join(email_feats)
        .join(file_feats)
        .fillna(0)
        .reset_index()
    )

    # This is the label-free table Day 4 models will actually train on.
    features.to_csv(f"{DATA_DIR}/user_features.csv", index=False)

    # Preview-only copy with labels attached, for our own inspection.
    preview = features.merge(labels, on="user_id", how="left")
    preview.to_csv(f"{DATA_DIR}/user_features_preview_with_labels.csv", index=False)

    print(f"Engineered {features.shape[1]-1} features for {features.shape[0]} users.")
    print(f"Feature columns: {[c for c in features.columns if c != 'user_id']}")
    return features, preview


if __name__ == "__main__":
    main()
