"""
Optional loader for the CERT Insider Threat Test Dataset (r4.2, CMU/DARPA).

This project's primary dataset is the synthetic generator in
generate_logs.py / inject_anomalies.py, which already satisfies
Objective 1. This loader exists for the OPTIONAL cross-validation
step mentioned in the proposal/interim report, in case you later want
to sanity-check model behaviour against the published CERT dataset too.

The dataset itself is NOT included in this repo (several GB, not
suitable for git). To use this loader:

  1. Download "Insider Threat Test Dataset" release r4.2 from CMU's
     Figshare/kilthub repository, or a mirror on Kaggle.
  2. Extract it and place the raw CSVs (logon.csv, device.csv, http.csv,
     email.csv, file.csv, psychometric.csv, LDAP/*.csv) under:
         data/raw/cert_r4.2/
  3. Run this script - it maps CERT's native columns onto the same
     schema used by our synthetic dataset (user_id, pc, activity/action,
     timestamp, ...) so the Day 3+ feature engineering code can consume
     either source interchangeably.

CERT r4.2 also ships an `answers/` directory with ground-truth insider
scenario logs (r4.2-1.csv ... r4.2-5.csv) - these map onto the same
role as our labels.csv and should be loaded via load_cert_answers().
"""

import glob
import os

import pandas as pd

CERT_DIR = "data/raw/cert_r4.2"


def _require_dir():
    if not os.path.isdir(CERT_DIR):
        raise FileNotFoundError(
            f"'{CERT_DIR}' not found. Download CERT r4.2 manually and place "
            f"the raw files there first (see module docstring)."
        )


def load_cert_logon():
    _require_dir()
    df = pd.read_csv(f"{CERT_DIR}/logon.csv")
    # CERT native columns: id,date,user,pc,activity
    df = df.rename(columns={"user": "user_id", "date": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df[["user_id", "pc", "activity", "timestamp"]]


def load_cert_device():
    _require_dir()
    df = pd.read_csv(f"{CERT_DIR}/device.csv")
    df = df.rename(columns={"user": "user_id", "date": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df[["user_id", "pc", "activity", "timestamp"]]


def load_cert_email():
    _require_dir()
    df = pd.read_csv(f"{CERT_DIR}/email.csv")
    # CERT native columns include: id,date,user,pc,to,cc,bcc,from,size,attachments
    df = df.rename(columns={"user": "user_id", "date": "timestamp", "size": "size_kb"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["external"] = ~df["to"].fillna("").str.contains("dtaa.com", case=False)
    return df[["user_id", "to", "external", "size_kb", "attachments", "timestamp"]]


def load_cert_file():
    _require_dir()
    df = pd.read_csv(f"{CERT_DIR}/file.csv")
    df = df.rename(columns={"user": "user_id", "date": "timestamp", "filename": "filename"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "activity" in df.columns:
        df = df.rename(columns={"activity": "action"})
    return df[["user_id", "pc", "filename", "action", "timestamp"]]


def load_cert_answers():
    """Loads CERT's ground-truth malicious-insider scenario logs (answers/ dir)
    into a labels.csv-compatible DataFrame: user_id, is_malicious, scenario."""
    _require_dir()
    answer_files = glob.glob(f"{CERT_DIR}/answers/*.csv")
    if not answer_files:
        raise FileNotFoundError(
            f"No answer files found under {CERT_DIR}/answers/. These ship "
            "separately with the CERT release as the ground-truth insider list."
        )
    frames = []
    for i, path in enumerate(sorted(answer_files)):
        df = pd.read_csv(path)
        user_col = "user" if "user" in df.columns else df.columns[0]
        scenario_users = df[user_col].dropna().unique()
        frames.append(
            pd.DataFrame(
                {
                    "user_id": scenario_users,
                    "is_malicious": True,
                    "scenario": f"cert_scenario_{i+1}",
                }
            )
        )
    return pd.concat(frames, ignore_index=True).drop_duplicates("user_id")


if __name__ == "__main__":
    _require_dir()
    print("CERT r4.2 raw files found - loading a quick preview...")
    print(load_cert_logon().head())
