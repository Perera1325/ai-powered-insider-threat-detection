"""
Day 1 (part 2) - Red-team anomaly injection.

Reads the normal baseline logs produced by generate_logs.py, picks a small
subset of users to act as "malicious insiders", and injects labeled
anomalous behaviour patterns into the logon/device/http/email/file logs:

    1. after_hours_login   - logins at unusual times (e.g. 2-4am)
    2. mass_file_access    - a burst of file open/copy events in a short window
    3. usb_exfiltration    - unusual USB connect + burst of file copy activity
    4. data_exfil_email    - large attachment sent to an external address
    5. privilege_probe     - repeated access to file paths outside the user's dept

Ground truth labels are written to data/processed/full/labels.csv and are
kept ONLY for evaluation - they must never be used as a model input.
"""

import random
from datetime import timedelta

import numpy as np
import pandas as pd

random.seed(7)
np.random.seed(7)

SCENARIOS = [
    "after_hours_login",
    "mass_file_access",
    "usb_exfiltration",
    "data_exfil_email",
    "privilege_probe",
]

NUM_MALICIOUS_USERS = 6
DATA_DIR = "data/processed/full"


def load_logs():
    users = pd.read_csv(f"{DATA_DIR}/users.csv")
    logon = pd.read_csv(f"{DATA_DIR}/logon.csv", parse_dates=["timestamp"])
    device = pd.read_csv(f"{DATA_DIR}/device.csv", parse_dates=["timestamp"])
    http = pd.read_csv(f"{DATA_DIR}/http.csv", parse_dates=["timestamp"])
    email = pd.read_csv(f"{DATA_DIR}/email.csv", parse_dates=["timestamp"])
    file_df = pd.read_csv(f"{DATA_DIR}/file.csv", parse_dates=["timestamp"])
    return users, logon, device, http, email, file_df


def inject_after_hours_login(logon, user_id, day):
    ts = day.replace(hour=random.choice([1, 2, 3, 4]), minute=random.randint(0, 59))
    new_row = {"user_id": user_id, "pc": f"PC-{user_id}", "activity": "Logon", "timestamp": ts}
    return pd.concat([logon, pd.DataFrame([new_row])], ignore_index=True)


def inject_mass_file_access(file_df, user_id, pc, day):
    base = day.replace(hour=23, minute=0)
    rows = []
    for i in range(random.randint(40, 80)):
        rows.append(
            {
                "user_id": user_id,
                "pc": pc,
                "filename": f"\\\\fileserver\\shared\\finance\\confidential_{i}.xlsx",
                "action": "Copy",
                "timestamp": base + timedelta(seconds=i * 20),
            }
        )
    return pd.concat([file_df, pd.DataFrame(rows)], ignore_index=True)


def inject_usb_exfiltration(device, file_df, user_id, pc, day):
    connect_time = day.replace(hour=22, minute=30)
    disconnect_time = connect_time + timedelta(minutes=45)
    new_rows = pd.DataFrame(
        [
            {"user_id": user_id, "pc": pc, "activity": "Connect", "timestamp": connect_time},
            {"user_id": user_id, "pc": pc, "activity": "Disconnect", "timestamp": disconnect_time},
        ]
    )
    device = pd.concat([device, new_rows], ignore_index=True)
    copy_rows = []
    for i in range(random.randint(20, 50)):
        copy_rows.append(
            {
                "user_id": user_id,
                "pc": pc,
                "filename": f"C:\\Users\\{user_id}\\Documents\\project_{i}.docx",
                "action": "Copy",
                "timestamp": connect_time + timedelta(minutes=i),
            }
        )
    file_df = pd.concat([file_df, pd.DataFrame(copy_rows)], ignore_index=True)
    return device, file_df


def inject_data_exfil_email(email, user_id, day):
    ts = day.replace(hour=random.randint(18, 21))
    new_row = {
        "user_id": user_id,
        "to": f"personal.account{random.randint(1,99)}@gmail.com",
        "external": True,
        "size_kb": random.randint(15000, 40000),
        "attachments": random.randint(3, 8),
        "timestamp": ts,
    }
    return pd.concat([email, pd.DataFrame([new_row])], ignore_index=True)


def inject_privilege_probe(file_df, user_id, pc, day, foreign_dept):
    rows = []
    for i in range(random.randint(15, 30)):
        rows.append(
            {
                "user_id": user_id,
                "pc": pc,
                "filename": f"\\\\fileserver\\shared\\{foreign_dept.lower()}\\restricted_{i}.pdf",
                "action": "Open",
                "timestamp": day.replace(hour=random.randint(9, 17), minute=random.randint(0, 59)),
            }
        )
    return pd.concat([file_df, pd.DataFrame(rows)], ignore_index=True)


def main():
    users, logon, device, http, email, file_df = load_logs()

    malicious_users = users.sample(n=NUM_MALICIOUS_USERS, random_state=7)
    dates = pd.to_datetime(logon["timestamp"]).dt.normalize().unique()
    dates = pd.Series(dates).sort_values().tolist()

    labels = []
    for i, (_, u) in enumerate(malicious_users.iterrows()):
        scenario = SCENARIOS[i % len(SCENARIOS)]
        day = random.choice(dates[5:-2])  # avoid very first/last days

        if scenario == "after_hours_login":
            logon = inject_after_hours_login(logon, u.user_id, day)
        elif scenario == "mass_file_access":
            file_df = inject_mass_file_access(file_df, u.user_id, u.pc, day)
        elif scenario == "usb_exfiltration":
            device, file_df = inject_usb_exfiltration(device, file_df, u.user_id, u.pc, day)
        elif scenario == "data_exfil_email":
            email = inject_data_exfil_email(email, u.user_id, day)
        elif scenario == "privilege_probe":
            other_depts = [d for d in users.department.unique() if d != u.department]
            file_df = inject_privilege_probe(file_df, u.user_id, u.pc, day, random.choice(other_depts))

        labels.append(
            {
                "user_id": u.user_id,
                "is_malicious": True,
                "scenario": scenario,
                "injected_day": str(day.date()),
            }
        )

    # ground truth for ALL users (non-malicious explicitly marked False)
    labels_df = pd.DataFrame(labels)  # only the malicious users, with scenario + day
    all_labels = users[["user_id"]].copy().merge(labels_df, on="user_id", how="left")
    all_labels["is_malicious"] = all_labels["is_malicious"].apply(lambda v: bool(v) if pd.notna(v) else False)

    logon.sort_values("timestamp").to_csv(f"{DATA_DIR}/logon.csv", index=False)
    device.sort_values("timestamp").to_csv(f"{DATA_DIR}/device.csv", index=False)
    email.sort_values("timestamp").to_csv(f"{DATA_DIR}/email.csv", index=False)
    file_df.sort_values("timestamp").to_csv(f"{DATA_DIR}/file.csv", index=False)
    all_labels.to_csv(f"{DATA_DIR}/labels.csv", index=False)

    print(f"Injected {len(labels)} red-team insider scenarios:")
    for l in labels:
        print(f"  {l['user_id']}: {l['scenario']} on {l['injected_day']}")
    print(f"Ground truth written to {DATA_DIR}/labels.csv")


if __name__ == "__main__":
    main()
