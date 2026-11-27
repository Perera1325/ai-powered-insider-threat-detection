"""SQLite persistence for case history, the audit log and the notification feed.

One small file database (default data/runtime/itd.db, override with ITD_DB). Survives page refreshes and
app restarts on a normal machine. On Streamlit Community Cloud the container disk is ephemeral: data
survives refreshes but is lost when the app is rebuilt, so the CSV case-log export remains the durable copy.
"""
import os
import sqlite3
from contextlib import closing
from datetime import datetime

import pandas as pd

import soc_tools as soc

DEFAULT_DB = os.path.join("data", "runtime", "itd.db")
SCHEMA = """
CREATE TABLE IF NOT EXISTS case_history(id INTEGER PRIMARY KEY, user_id TEXT NOT NULL, time TEXT NOT NULL,
    status TEXT NOT NULL, note TEXT, analyst TEXT);
CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY, time TEXT NOT NULL, actor TEXT, role TEXT,
    action TEXT NOT NULL, detail TEXT);
CREATE TABLE IF NOT EXISTS notifications(id INTEGER PRIMARY KEY, time TEXT NOT NULL, severity TEXT,
    title TEXT NOT NULL, message TEXT, user_id TEXT, source TEXT, delivered TEXT, dedupe_key TEXT UNIQUE);
"""


def db_path():
    return os.environ.get("ITD_DB", DEFAULT_DB)


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def connect(path=None):
    path = path or db_path()
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    con = sqlite3.connect(path, timeout=5)
    con.executescript(SCHEMA)
    return con


# ------------------------------------------------------------------ cases
def add_case_entry(uid, status, note="", analyst="analyst", path=None, time=None):
    with closing(connect(path)) as con, con:
        con.execute("INSERT INTO case_history(user_id,time,status,note,analyst) VALUES(?,?,?,?,?)",
                    (uid, time or _now(), status, (note or "").strip(), analyst))


def load_store(path=None):
    with closing(connect(path)) as con:
        df = pd.read_sql_query("SELECT user_id,time,status,note,analyst FROM case_history ORDER BY id", con)
    return soc.cases_from_frame(df)


def replace_cases(store, path=None):
    """Replace all stored cases (used when a CSV case log is restored)."""
    df = soc.cases_to_frame(store)
    with closing(connect(path)) as con, con:
        con.execute("DELETE FROM case_history")
        con.executemany("INSERT INTO case_history(user_id,time,status,note,analyst) VALUES(?,?,?,?,?)",
                        [(r.user_id, r.time, r.status, r.note, r.analyst) for r in df.itertuples(index=False)])


# ------------------------------------------------------------------ audit
def log_audit(actor, role, action, detail="", path=None):
    with closing(connect(path)) as con, con:
        con.execute("INSERT INTO audit(time,actor,role,action,detail) VALUES(?,?,?,?,?)",
                    (_now(), actor, role, action, str(detail)[:500]))


def audit_frame(limit=500, path=None):
    with closing(connect(path)) as con:
        return pd.read_sql_query("SELECT time,actor,role,action,detail FROM audit ORDER BY id DESC LIMIT ?", con, params=(limit,))


# ------------------------------------------------------------------ notifications
def add_notification(title, message="", severity="High", user_id="", source="system", delivered="feed",
                     dedupe_key=None, path=None):
    """Insert a notification; returns True if new, False if dedupe_key already exists."""
    try:
        with closing(connect(path)) as con, con:
            con.execute("INSERT INTO notifications(time,severity,title,message,user_id,source,delivered,dedupe_key) "
                        "VALUES(?,?,?,?,?,?,?,?)", (_now(), severity, title, message, user_id, source, delivered, dedupe_key))
        return True
    except sqlite3.IntegrityError:
        return False


def notifications_frame(limit=200, path=None):
    with closing(connect(path)) as con:
        return pd.read_sql_query("SELECT time,severity,title,message,user_id,source,delivered FROM notifications "
                                 "ORDER BY id DESC LIMIT ?", con, params=(limit,))


def clear_notifications(path=None):
    with closing(connect(path)) as con, con:
        con.execute("DELETE FROM notifications")
