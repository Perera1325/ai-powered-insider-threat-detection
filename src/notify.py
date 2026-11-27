"""Alert notifications: in-app feed (SQLite) plus an optional Slack-compatible webhook.

The webhook URL is read ONLY from configuration (ITD_WEBHOOK_URL env or Streamlit secret `webhook_url`),
never from a user-editable field, and must be https - so a dashboard user cannot make the server call
arbitrary addresses.
"""
import json
import os
import urllib.request
from urllib.parse import urlparse

import case_store as cs


def webhook_url(secret=None):
    url = (secret or os.environ.get("ITD_WEBHOOK_URL") or "").strip()
    return url if urlparse(url).scheme == "https" and urlparse(url).netloc else ""


def post_webhook(url, text, timeout=3):
    """Returns (ok, message). Never raises."""
    if not url:
        return False, "no webhook configured"
    req = urllib.request.Request(url, data=json.dumps({"text": text}).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (https enforced above)
            return 200 <= r.status < 300, f"HTTP {r.status}"
    except Exception as e:  # network errors must not break the dashboard
        return False, type(e).__name__


def notify_alert(alert, techniques="", secret_url=None, path=None):
    """Create a feed entry (and webhook post) for a real-time alert dict. Returns True if newly raised."""
    key = f"rt:{alert.get('stream', '')}:{alert.get('tick')}:{alert['user_id']}"
    title = f"HIGH RISK: {alert['user_id']} (risk {alert['risk_score']:.0f}/100)"
    msg = (f"Simulated day {alert.get('tick')}, detected in {alert.get('latency_s', 0):.2f} s. "
           f"Drivers: {alert.get('drivers', '-') or '-'}." + (f" ATT&CK: {techniques}." if techniques else ""))
    url = webhook_url(secret_url)
    new = cs.add_notification(title, msg, "High", alert["user_id"], "real-time", "feed", dedupe_key=key, path=path)
    if new and url:
        ok, info = post_webhook(url, f"{title}\n{msg}")
        with cs.closing(cs.connect(path)) as con, con:
            con.execute("UPDATE notifications SET delivered=? WHERE dedupe_key=?",
                        ("feed+webhook" if ok else f"feed (webhook failed: {info})", key))
    return new


def notify_case(uid, status, analyst, note="", secret_url=None, path=None):
    """Escalations are pushed to the feed/webhook; other status changes are only audited."""
    if status != "Escalated":
        return False
    key = f"case:{uid}:{cs._now()}"
    title = f"Case ESCALATED: {uid}"
    msg = f"Escalated by {analyst}." + (f" Note: {note}" if note else "")
    new = cs.add_notification(title, msg, "High", uid, "case", "feed", dedupe_key=key, path=path)
    url = webhook_url(secret_url)
    if new and url:
        post_webhook(url, f"{title}\n{msg}")
    return new
