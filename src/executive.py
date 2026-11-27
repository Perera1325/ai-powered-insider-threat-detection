"""Executive overview numbers: case funnel, time to resolve, SLA health and a one-page summary.

Pure functions, no Streamlit."""
from datetime import datetime

import pandas as pd

import playbook
import soc_tools as soc


def case_funnel(store):
    """Cases per status (every status listed, so the chart always has the same bars)."""
    counts = {s: 0 for s in soc.STATUSES}
    for c in store.values():
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    return pd.DataFrame({"status": list(counts), "cases": list(counts.values())})


def resolution_hours(store):
    """Hours from the first analyst action to closure, for every closed case."""
    out = {}
    for uid, c in store.items():
        if c["status"] in playbook.CLOSED and c["history"]:
            out[uid] = (playbook._parse(c["history"][-1]["time"]) - playbook._parse(c["history"][0]["time"])).total_seconds() / 3600
    return out


def mean_time_to_resolve(store):
    r = resolution_hours(store)
    return round(sum(r.values()) / len(r), 1) if r else None


def executive_kpis(alert_df, store, sla):
    """Headline numbers for the overview tab. sla = playbook.sla_table(...)."""
    in_scope = alert_df.user_id.tolist()
    sc = {u: store[u] for u in in_scope if u in store}
    closed = [u for u, c in sc.items() if c["status"] in playbook.CLOSED]
    return {
        "alerts": len(alert_df),
        "high": int((alert_df.risk_level == "High").sum()),
        "untriaged_high": int(((sla.risk_level == "High") & (sla.state == "Untriaged")).sum()),
        "open_cases": int(sum(1 for c in sc.values() if c["status"] in soc.OPEN_STATUSES)),
        "escalated": int(sum(1 for c in sc.values() if c["status"] == "Escalated")),
        "closed": len(closed),
        "sla_breached": int((sla.state == "Breached").sum()),
        "sla_at_risk": int((sla.state == "At risk").sum()),
        "mttr_hours": mean_time_to_resolve(sc),
    }


def executive_summary(kpis, top_users, departments, model_name, generated=None):
    """One-page Markdown summary for a manager. top_users: frame(user_id, risk_score, risk_level, case_status);
    departments: risk_intel.department_risk output."""
    generated = (generated or datetime.now()).strftime("%Y-%m-%d %H:%M")
    mttr = "n/a" if kpis["mttr_hours"] is None else f"{kpis['mttr_hours']} h"
    lines = [f"# Insider threat posture - executive summary", "",
             f"*Generated {generated}. Model: {model_name}. Scores are anomaly indicators, not proof of intent.*", "",
             "## Headlines",
             f"- {kpis['alerts']} alerts in the queue, {kpis['high']} high severity ({kpis['untriaged_high']} not yet triaged)",
             f"- {kpis['open_cases']} open cases ({kpis['escalated']} escalated), {kpis['closed']} closed",
             f"- Response SLA: {kpis['sla_breached']} breached, {kpis['sla_at_risk']} at risk; mean time to resolve {mttr}", "",
             "## Highest-risk entities"]
    for r in top_users.head(5).itertuples(index=False):
        lines.append(f"- {r.user_id}: {r.risk_score:.1f} ({r.risk_level}), case {r.case_status}")
    lines += ["", "## Departments with the most high-risk users"]
    for r in departments.head(3).itertuples(index=False):
        lines.append(f"- {r.department}: {r.high_risk_users} of {r.users} users above the alert threshold (mean risk {r.mean_risk})")
    return "\n".join(lines) + "\n"
