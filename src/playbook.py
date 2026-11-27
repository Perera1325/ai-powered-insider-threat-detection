"""Response playbooks, case SLAs and incident briefs (SOAR-style guidance for the analyst).

Pure functions, no Streamlit. A playbook is a checklist of response steps chosen from the alert's severity
and the MITRE ATT&CK techniques suggested by the user's behaviour. It is guidance for a human analyst: the
dashboard never takes an action on an account by itself.
"""
from datetime import datetime, timedelta

import pandas as pd

import mitre

# time allowed from the first analyst action on a case to its closure
SLA_HOURS = {"High": 4, "Medium": 24, "Low": 72}
AT_RISK_SHARE = 0.25  # "At risk" once less than 25% of the SLA window is left
CLOSED = ("Closed - false positive", "Closed - confirmed")
PHASES = ["Triage", "Contain", "Investigate", "Recover"]

# step = (phase, action, owner)
SEVERITY_STEPS = {
    "High": [
        ("Triage", "Acknowledge the alert and open a case within the SLA window", "SOC analyst"),
        ("Contain", "Ask IT to enforce step-up authentication and review the account's active sessions", "SOC lead"),
        ("Contain", "Snapshot the user's recent logon, file, e-mail and device logs as evidence", "SOC analyst"),
        ("Investigate", "Brief the HR / insider-risk contact before any contact with the user", "SOC lead"),
        ("Recover", "Record the outcome, update detection thresholds and close the case with a note", "SOC analyst"),
    ],
    "Medium": [
        ("Triage", "Review the top behavioural deviations and compare with the user's department", "SOC analyst"),
        ("Investigate", "Check whether a business reason explains the change (project, travel, new role)", "SOC analyst"),
        ("Recover", "Close as false positive or escalate to High with a note", "SOC analyst"),
    ],
    "Low": [
        ("Triage", "No action needed; keep the user on the watch list", "SOC analyst"),
        ("Recover", "Re-check at the next review cycle", "SOC analyst"),
    ],
}

TECHNIQUE_STEPS = {
    "T1078": [("Investigate", "Compare logon times and source machines with the user's normal pattern; check for shared or stolen credentials", "SOC analyst"),
              ("Contain", "Force a password reset if the logons cannot be explained", "IT service desk")],
    "T1039": [("Investigate", "List the shared folders touched and check the user's access need for each one", "SOC analyst"),
              ("Contain", "Review and tighten folder permissions that the user does not need", "File server admin")],
    "T1005": [("Investigate", "Review which local files were copied and whether they contain confidential data", "SOC analyst")],
    "T1083": [("Investigate", "Check whether directory browsing matches the user's job (search for enumeration of unrelated folders)", "SOC analyst")],
    "T1069": [("Investigate", "Review which groups and shares the user probed; look for access they do not hold", "SOC analyst")],
    "T1052.001": [("Contain", "Block removable media for the user's workstation until reviewed", "Endpoint admin"),
                  ("Investigate", "Pull the USB device log and the files copied around each connection", "SOC analyst")],
    "T1048": [("Investigate", "Review the size, recipients and attachments of external e-mails; check for personal addresses", "SOC analyst"),
              ("Contain", "Add a temporary DLP rule for large external attachments from this user", "Mail admin")],
}


def sla_hours(severity):
    return SLA_HOURS.get(severity, SLA_HOURS["Low"])


def build_playbook(severity, techniques=()):
    """Ordered response checklist for one alert. techniques: ATT&CK ids (e.g. ['T1078'])."""
    steps = [(p, a, o, "Severity: " + severity) for p, a, o in SEVERITY_STEPS.get(severity, SEVERITY_STEPS["Low"])]
    if severity != "Low":  # technique-specific steps only matter when the alert is worth working
        for t in techniques:
            steps += [(p, a, o, t) for p, a, o in TECHNIQUE_STEPS.get(t, [])]
    seen, rows = set(), []
    for p, a, o, src in sorted(steps, key=lambda s: PHASES.index(s[0])):  # sorted() is stable: order kept inside a phase
        if a in seen:
            continue
        seen.add(a)
        rows.append({"phase": p, "action": a, "owner": o, "why": src})
    df = pd.DataFrame(rows, columns=["phase", "action", "owner", "why"])
    df.insert(0, "step", range(1, len(df) + 1))
    return df


def _parse(ts):
    return datetime.strptime(str(ts), "%Y-%m-%d %H:%M:%S")


def case_sla(case, severity, now=None):
    """SLA state of one case dict ({'status', 'history'}). Clock starts at the first analyst action."""
    now = now or datetime.now()
    hist = (case or {}).get("history", [])
    window = timedelta(hours=sla_hours(severity))
    if not hist:
        return {"state": "Untriaged", "started": None, "due": None, "hours_left": None, "window_hours": sla_hours(severity)}
    started = _parse(hist[0]["time"])
    due = started + window
    status = case["status"]
    if status in CLOSED:
        closed_at = _parse(hist[-1]["time"])
        return {"state": "Met" if closed_at <= due else "Breached", "started": started, "due": due,
                "hours_left": None, "window_hours": sla_hours(severity)}
    left = (due - now).total_seconds() / 3600
    if left < 0:
        state = "Breached"
    elif left < AT_RISK_SHARE * sla_hours(severity):
        state = "At risk"
    else:
        state = "On track"
    return {"state": state, "started": started, "due": due, "hours_left": round(left, 1), "window_hours": sla_hours(severity)}


def sla_table(alert_df, store, now=None):
    rows = []
    for r in alert_df.itertuples(index=False):
        s = case_sla(store.get(r.user_id), r.risk_level, now)
        rows.append({"user_id": r.user_id, "risk_level": r.risk_level, "sla_hours": s["window_hours"], "state": s["state"],
                     "started": s["started"].strftime("%Y-%m-%d %H:%M") if s["started"] else "-",
                     "due": s["due"].strftime("%Y-%m-%d %H:%M") if s["due"] else "-",
                     "hours_left": s["hours_left"]})
    return pd.DataFrame(rows, columns=["user_id", "risk_level", "sla_hours", "state", "started", "due", "hours_left"])


def incident_brief(uid, risk_score, severity, case_status, matches, deviations, playbook, generated=None):
    """Plain-language incident brief (Markdown) an analyst can paste into a ticket."""
    generated = (generated or datetime.now()).strftime("%Y-%m-%d %H:%M")
    lines = [f"# Incident brief: {uid}", "",
             f"*Generated {generated}. Risk is an unsupervised anomaly score, not proof of malicious intent.*", "",
             f"- **Risk score:** {risk_score:.1f}/100 ({severity})",
             f"- **Case status:** {case_status}",
             f"- **Response SLA:** {sla_hours(severity)} hours from the first analyst action", "",
             "## Why this user was flagged"]
    if len(deviations):
        for r in deviations.itertuples(index=False):
            lines.append(f"- {r.feature}: {r.value:.2f} against a population mean of {r.population_mean:.2f} ({r.z_score:+.1f} sd)")
    else:
        lines.append("- No single behaviour stands out strongly.")
    lines += ["", "## Suggested ATT&CK techniques (heuristic)"]
    if len(matches):
        for r in matches.itertuples(index=False):
            lines.append(f"- [{r.technique} {r.name}]({mitre.technique_url(r.technique)}) ({r.tactic})")
    else:
        lines.append("- None suggested.")
    lines += ["", "## Response checklist"]
    for r in playbook.itertuples(index=False):
        lines.append(f"{r.step}. **{r.phase}** - {r.action} _(owner: {r.owner})_")
    return "\n".join(lines) + "\n"
