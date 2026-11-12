"""
Day 1 - Synthetic organizational activity log generator.

Generates realistic (but fake) organizational activity data for the
AI-Powered Insider Threat Detection project:
    - logon.csv   : user login/logout events
    - device.csv  : USB / removable device connect events
    - http.csv    : web browsing activity
    - email.csv   : email send metadata (sender, recipients, size, attachments)
    - file.csv    : file access events (open/copy/delete)
    - users.csv   : simulated org directory (role, department, PC)

No real personal or company data is used anywhere in this project.
"""

import random
import string
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)

NUM_USERS = 100
NUM_DAYS = 30
START_DATE = datetime(2026, 6, 1)

DEPARTMENTS = ["Finance", "Engineering", "HR", "Sales", "IT", "Legal", "Marketing"]
ROLES = ["Employee", "Manager", "Contractor", "Admin"]
DOMAINS = ["gmail.com", "yahoo.com", "outlook.com"]  # for "external" email recipients
INTERNAL_DOMAIN = "simcorp-internal.local"


def generate_users(num_users=NUM_USERS):
    users = []
    for i in range(num_users):
        uid = f"U{i+1:04d}"
        dept = random.choice(DEPARTMENTS)
        role = random.choices(ROLES, weights=[70, 15, 10, 5])[0]
        users.append(
            {
                "user_id": uid,
                "full_name": fake.name(),
                "department": dept,
                "role": role,
                "pc": f"PC-{dept[:3].upper()}-{i+1:03d}",
                "email": f"{uid.lower()}@{INTERNAL_DOMAIN}",
            }
        )
    return pd.DataFrame(users)


def _safe_df(rows, columns):
    """Returns a properly-columned (possibly empty) DataFrame, sorted by
    timestamp. Guards against the case where `rows` is empty - a plain
    `pd.DataFrame([])` has NO columns at all, which breaks
    `.sort_values("timestamp")`. This matters for small user cohorts /
    single-day windows (e.g. the live demo module) where it's entirely
    possible a given day produces zero events of some type - with the
    full 100-user/30-day baseline dataset this edge case never came up.
    """
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def _business_hour_ts(day):
    """Random timestamp within typical 08:00-18:00 business hours."""
    hour = np.random.normal(loc=13, scale=2.5)
    hour = min(max(hour, 7), 19)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return day.replace(hour=int(hour), minute=minute, second=second)


def generate_logon(users_df, num_days=NUM_DAYS, start_date=START_DATE):
    rows = []
    for day_offset in range(num_days):
        day = start_date + timedelta(days=day_offset)
        if day.weekday() >= 5:  # skip weekends for normal behaviour
            continue
        for _, u in users_df.iterrows():
            if random.random() < 0.05:  # ~5% absent that day
                continue
            logon_time = _business_hour_ts(day)
            session_len = max(15, np.random.normal(loc=420, scale=90))  # minutes
            logoff_time = logon_time + timedelta(minutes=session_len)
            rows.append(
                {
                    "user_id": u.user_id,
                    "pc": u.pc,
                    "activity": "Logon",
                    "timestamp": logon_time,
                }
            )
            rows.append(
                {
                    "user_id": u.user_id,
                    "pc": u.pc,
                    "activity": "Logoff",
                    "timestamp": logoff_time,
                }
            )
    return _safe_df(rows, ["user_id", "pc", "activity", "timestamp"])


def generate_device(users_df, num_days=NUM_DAYS, start_date=START_DATE):
    rows = []
    for day_offset in range(num_days):
        day = start_date + timedelta(days=day_offset)
        if day.weekday() >= 5:
            continue
        for _, u in users_df.iterrows():
            # most users rarely use removable devices
            if random.random() < 0.06:
                connect_time = _business_hour_ts(day)
                disconnect_time = connect_time + timedelta(minutes=random.randint(2, 30))
                rows.append(
                    {"user_id": u.user_id, "pc": u.pc, "activity": "Connect", "timestamp": connect_time}
                )
                rows.append(
                    {"user_id": u.user_id, "pc": u.pc, "activity": "Disconnect", "timestamp": disconnect_time}
                )
    return _safe_df(rows, ["user_id", "pc", "activity", "timestamp"])


def generate_http(users_df, num_days=NUM_DAYS, start_date=START_DATE):
    sites = [
        "intranet.simcorp-internal.local", "sharepoint.simcorp-internal.local",
        "github.com", "stackoverflow.com", "news.example.com", "mail.google.com",
        "linkedin.com", "docs.google.com",
    ]
    rows = []
    for day_offset in range(num_days):
        day = start_date + timedelta(days=day_offset)
        if day.weekday() >= 5:
            continue
        for _, u in users_df.iterrows():
            for _ in range(np.random.poisson(6)):
                rows.append(
                    {
                        "user_id": u.user_id,
                        "pc": u.pc,
                        "url": random.choice(sites),
                        "timestamp": _business_hour_ts(day),
                    }
                )
    return _safe_df(rows, ["user_id", "pc", "url", "timestamp"])


def generate_email(users_df, num_days=NUM_DAYS, start_date=START_DATE):
    user_ids = users_df.user_id.tolist()
    rows = []
    for day_offset in range(num_days):
        day = start_date + timedelta(days=day_offset)
        if day.weekday() >= 5:
            continue
        for _, u in users_df.iterrows():
            for _ in range(np.random.poisson(3)):
                num_recipients = random.randint(1, 4)
                recipients = random.sample(user_ids, k=min(num_recipients, len(user_ids)))
                is_external = random.random() < 0.08
                to = (
                    [f"contact{random.randint(1,999)}@{random.choice(DOMAINS)}"]
                    if is_external
                    else [f"{r.lower()}@{INTERNAL_DOMAIN}" for r in recipients]
                )
                rows.append(
                    {
                        "user_id": u.user_id,
                        "to": ";".join(to),
                        "external": is_external,
                        "size_kb": max(1, int(np.random.exponential(120))),
                        "attachments": np.random.poisson(0.3),
                        "timestamp": _business_hour_ts(day),
                    }
                )
    return _safe_df(rows, ["user_id", "to", "external", "size_kb", "attachments", "timestamp"])


def generate_file(users_df, num_days=NUM_DAYS, start_date=START_DATE):
    actions = ["Open", "Copy", "Write", "Delete"]
    weights = [70, 15, 10, 5]
    file_paths = [
        "\\\\fileserver\\shared\\reports\\", "\\\\fileserver\\shared\\hr\\",
        "\\\\fileserver\\shared\\finance\\", "\\\\fileserver\\shared\\projects\\",
        "C:\\Users\\{}\\Documents\\",
    ]
    rows = []
    for day_offset in range(num_days):
        day = start_date + timedelta(days=day_offset)
        if day.weekday() >= 5:
            continue
        for _, u in users_df.iterrows():
            for _ in range(np.random.poisson(8)):
                path = random.choice(file_paths).format(u.user_id)
                fname = f"{fake.word()}_{random.randint(1,999)}.{random.choice(['docx','xlsx','pdf','txt'])}"
                rows.append(
                    {
                        "user_id": u.user_id,
                        "pc": u.pc,
                        "filename": path + fname,
                        "action": random.choices(actions, weights=weights)[0],
                        "timestamp": _business_hour_ts(day),
                    }
                )
    return _safe_df(rows, ["user_id", "pc", "filename", "action", "timestamp"])


def reseed(seed):
    """Re-seed every RNG used by the generator (for multi-seed experiments)."""
    Faker.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def main(out_dir="data/processed/full", seed=None, num_users=NUM_USERS):
    import os
    if seed is not None:
        reseed(seed)
    os.makedirs(out_dir, exist_ok=True)
    users_df = generate_users(num_users)
    logon_df = generate_logon(users_df)
    device_df = generate_device(users_df)
    http_df = generate_http(users_df)
    email_df = generate_email(users_df)
    file_df = generate_file(users_df)

    users_df.to_csv(f"{out_dir}/users.csv", index=False)
    logon_df.to_csv(f"{out_dir}/logon.csv", index=False)
    device_df.to_csv(f"{out_dir}/device.csv", index=False)
    http_df.to_csv(f"{out_dir}/http.csv", index=False)
    email_df.to_csv(f"{out_dir}/email.csv", index=False)
    file_df.to_csv(f"{out_dir}/file.csv", index=False)

    print("Generated normal baseline activity:")
    print(f"  users   : {len(users_df)}")
    print(f"  logon   : {len(logon_df)} events")
    print(f"  device  : {len(device_df)} events")
    print(f"  http    : {len(http_df)} events")
    print(f"  email   : {len(email_df)} events")
    print(f"  file    : {len(file_df)} events")

    return users_df, logon_df, device_df, http_df, email_df, file_df


if __name__ == "__main__":
    main()
