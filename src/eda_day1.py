"""
Day 1 - Quick exploratory data analysis / sanity check on the generated
dataset. Confirms the data looks realistic and that the injected red-team
users are (at a glance) distinguishable from normal users on simple stats.
Run after generate_logs.py and inject_anomalies.py.
"""

import pandas as pd

DATA_DIR = "data/processed/full"


def main():
    logon = pd.read_csv(f"{DATA_DIR}/logon.csv", parse_dates=["timestamp"])
    device = pd.read_csv(f"{DATA_DIR}/device.csv", parse_dates=["timestamp"])
    http = pd.read_csv(f"{DATA_DIR}/http.csv", parse_dates=["timestamp"])
    email = pd.read_csv(f"{DATA_DIR}/email.csv", parse_dates=["timestamp"])
    file_df = pd.read_csv(f"{DATA_DIR}/file.csv", parse_dates=["timestamp"])
    labels = pd.read_csv(f"{DATA_DIR}/labels.csv")

    lines = []
    lines.append("# Day 1 EDA Summary\n")
    lines.append(f"- Date range: {logon.timestamp.min()} to {logon.timestamp.max()}")
    lines.append(f"- Users: {labels.user_id.nunique()}  (malicious: {int(labels.is_malicious.sum())})")
    lines.append(f"- Logon events: {len(logon)}")
    lines.append(f"- Device events: {len(device)}")
    lines.append(f"- HTTP events: {len(http)}")
    lines.append(f"- Email events: {len(email)}")
    lines.append(f"- File events: {len(file_df)}\n")

    lines.append("## Sanity check: malicious users stand out on simple aggregates\n")
    malicious_ids = labels[labels.is_malicious].user_id.tolist()

    file_counts = file_df.groupby("user_id").size()
    lines.append("File-access event count (top 10 users):")
    lines.append("```")
    lines.append(file_counts.sort_values(ascending=False).head(10).to_string())
    lines.append("```")
    lines.append(f"Malicious users: {malicious_ids}\n")

    logon_only = logon[logon.activity == "Logon"]
    after_hours = logon_only[(logon_only.timestamp.dt.hour < 6) | (logon_only.timestamp.dt.hour > 20)]
    lines.append("Logon (not logoff) events outside 06:00-20:00 (should be ~0 for normal users):")
    lines.append("```")
    lines.append(after_hours.groupby("user_id").size().sort_values(ascending=False).head(10).to_string() or "(none)")
    lines.append("```\n")

    external_email = email[email.external == True]  # noqa: E712
    lines.append("Users with largest external email attachments (KB):")
    lines.append("```")
    lines.append(
        external_email.sort_values("size_kb", ascending=False)[["user_id", "size_kb", "attachments"]]
        .head(10)
        .to_string(index=False)
    )
    lines.append("```")

    report = "\n".join(lines)
    with open("docs/day1_eda_summary.md", "w") as f:
        f.write(report + "\n")

    print(report)


if __name__ == "__main__":
    main()
