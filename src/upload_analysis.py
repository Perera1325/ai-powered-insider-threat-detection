"""Analyse user-supplied log files (CSV) with the same pipeline as the built-in dataset.

Two ways in:
  1. Raw logs  - any of logon.csv, device.csv, http.csv, email.csv, file.csv (+ optional users.csv)
  2. A ready feature table - a CSV with a user_id column and the 28 behavioural feature columns

Everything is validated first so a wrong file gives a clear message instead of a stack trace.
Scoring is unsupervised and fitted on the uploaded population itself (no labels needed).
"""
import io
import re

import numpy as np
import pandas as pd

import feature_engineering as fe

LOG_SCHEMAS = {
    "logon":  ["user_id", "activity", "timestamp"],
    "device": ["user_id", "activity", "timestamp"],
    "http":   ["user_id", "url", "timestamp"],
    "email":  ["user_id", "to", "external", "size_kb", "attachments", "timestamp"],
    "file":   ["user_id", "filename", "action", "timestamp"],
}
FEATURE_COLUMNS = [
    "total_logons", "avg_logon_hour", "std_logon_hour", "after_hours_logon_count", "after_hours_logon_ratio",
    "avg_session_min", "std_session_min", "device_connect_count", "device_after_hours_count",
    "http_event_count", "distinct_domains", "email_count", "avg_email_size_kb", "total_attachments",
    "external_email_count", "max_external_size_kb", "external_email_ratio", "avg_recipients", "max_recipients",
    "distinct_recipients", "file_event_count", "distinct_files", "copy_count", "delete_count", "copy_ratio",
    "max_daily_file_events", "shared_folder_categories_touched", "max_daily_shared_folder_events",
]
MIN_USERS = 20
MAX_ROWS = 600_000


def _read(source):
    if hasattr(source, "getvalue"):
        source = io.BytesIO(source.getvalue())
    elif isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    return pd.read_csv(source, nrows=MAX_ROWS)


def classify(filename):
    """Map a file name to a log type ('logon', 'device', ..., 'users', 'features') or None."""
    n = re.sub(r"[^a-z]", "", filename.lower().rsplit(".", 1)[0])
    for key in ("features", "users", "logon", "device", "http", "email", "file"):
        if key in n:
            return key
    return None


def load_logs(named_files):
    """named_files: dict {file name: file-like/bytes}. Returns (logs, users, features, messages)."""
    logs, users, features, msgs = {}, None, None, []
    for name, src in named_files.items():
        kind = classify(name)
        if kind is None:
            msgs.append(f"Skipped '{name}': file name should contain logon, device, http, email, file, users or features.")
            continue
        try:
            df = _read(src)
        except Exception as e:  # unreadable CSV
            raise ValueError(f"Could not read '{name}' as CSV ({e}).")
        if kind == "users":
            users = df
        elif kind == "features":
            features = df
        else:
            missing = [c for c in LOG_SCHEMAS[kind] if c not in df.columns]
            if missing:
                raise ValueError(f"'{name}' is missing column(s): {', '.join(missing)}. Expected: {', '.join(LOG_SCHEMAS[kind])}.")
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            bad = int(df["timestamp"].isna().sum())
            if bad:
                msgs.append(f"'{name}': dropped {bad} row(s) with unreadable timestamps.")
                df = df.dropna(subset=["timestamp"])
            if kind == "email":
                df["external"] = df["external"].astype(str).str.lower().isin(["true", "1", "yes"])
                df["size_kb"] = pd.to_numeric(df["size_kb"], errors="coerce").fillna(0)
                df["attachments"] = pd.to_numeric(df["attachments"], errors="coerce").fillna(0)
            logs[kind] = df
    return logs, users, features, msgs


def features_from_logs(logs, users=None):
    """Same feature engineering as the built-in dataset; missing log types give zero features."""
    ids = set()
    for df in logs.values():
        ids |= set(df["user_id"].astype(str))
    if users is not None and "user_id" in users.columns:
        ids |= set(users["user_id"].astype(str))
    for df in logs.values():
        df["user_id"] = df["user_id"].astype(str)
    if not ids:
        raise ValueError("No user_id values found in the uploaded logs.")
    builders = {"logon": fe.build_logon_features, "device": fe.build_device_features, "http": fe.build_http_features,
                "email": fe.build_email_features, "file": fe.build_file_features}
    out = pd.DataFrame(index=pd.Index(sorted(ids), name="user_id"))
    for kind, fn in builders.items():
        if kind in logs and len(logs[kind]):
            out = out.join(fn(logs[kind]))
    out = out.reindex(columns=FEATURE_COLUMNS).fillna(0).reset_index()
    return out


def features_from_table(df):
    if "user_id" not in df.columns:
        raise ValueError("Feature table needs a 'user_id' column.")
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    out = df.copy()
    out["user_id"] = out["user_id"].astype(str)
    out = out.drop_duplicates("user_id")
    for c in FEATURE_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0) if c in out.columns else 0.0
    return out[["user_id"] + FEATURE_COLUMNS], missing


def prepare(named_files):
    """Full ingest: returns (features_df, messages). Raises ValueError with a readable message."""
    logs, users, table, msgs = load_logs(named_files)
    if table is not None:
        feats, missing = features_from_table(table)
        if missing:
            msgs.append(f"Feature table lacked {len(missing)} column(s) ({', '.join(missing[:4])}...); filled with 0.")
    elif logs:
        feats = features_from_logs(logs, users)
        absent = [k for k in LOG_SCHEMAS if k not in logs]
        if absent:
            msgs.append("No " + ", ".join(absent) + " log uploaded; those features are 0 for everyone.")
    else:
        raise ValueError("Upload at least one log file (logon/device/http/email/file .csv) or one features .csv.")
    if len(feats) < MIN_USERS:
        raise ValueError(f"Found only {len(feats)} user(s); at least {MIN_USERS} are needed so 'normal' can be learned.")
    return feats, msgs


def results_table(features, models, model_name, X, du):
    """Ranked risk table for uploaded users (no ground truth)."""
    scores, risk = du.score_users(models, model_name, X)
    df = features[["user_id"]].copy()
    df["anomaly_score"] = scores
    df["risk_score"] = np.round(risk, 1)
    df["risk_level"] = df.risk_score.apply(du.risk_level)
    df = df.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df


def top_drivers(features, X, feature_cols, uid, n=3):
    """Cheap, instant explanation: the features where this user is furthest from the population (z-scores)."""
    i = int(np.where(features.user_id.values == uid)[0][0])
    z = pd.Series(X[i], index=feature_cols)
    return z.reindex(z.abs().sort_values(ascending=False).index).head(n)
