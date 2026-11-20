"""Analyst-workflow helpers for the dashboard: case management, user drill-down, threshold simulation,
multi-model agreement and PDF/CSV report export. Pure functions (no Streamlit) so they can be unit-tested."""
import io
from datetime import datetime

import numpy as np
import pandas as pd

STATUSES = ["New", "Investigating", "Escalated", "Closed - false positive", "Closed - confirmed"]
OPEN_STATUSES = ["Investigating", "Escalated"]  # "New" = not yet triaged


# ---------------------------------------------------------------- case management
def new_case_store():
    return {}


def update_case(store, uid, status, note="", analyst="analyst", now=None):
    """Append an audit entry and set the current status. Returns the case dict."""
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    now = (now or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    case = store.setdefault(uid, {"status": "New", "history": []})
    case["status"] = status
    case["history"].append({"time": now, "status": status, "note": note.strip(), "analyst": analyst})
    return case


def case_status(store, uid):
    return store.get(uid, {}).get("status", "New")


def latest_note(store, uid):
    hist = store.get(uid, {}).get("history", [])
    for h in reversed(hist):
        if h["note"]:
            return h["note"]
    return ""


def cases_to_frame(store):
    rows = [{"user_id": u, **h} for u, c in store.items() for h in c["history"]]
    cols = ["user_id", "time", "status", "analyst", "note"]
    return pd.DataFrame(rows, columns=cols)


def cases_from_frame(df):
    store = {}
    if df is None or df.empty or not {"user_id", "time", "status"}.issubset(df.columns):
        return store
    for r in df.sort_values("time").itertuples(index=False):
        if r.status not in STATUSES:
            continue
        c = store.setdefault(r.user_id, {"status": "New", "history": []})
        c["status"] = r.status
        c["history"].append({"time": str(r.time), "status": r.status,
                             "note": "" if pd.isna(getattr(r, "note", "")) else str(getattr(r, "note", "")),
                             "analyst": "analyst" if pd.isna(getattr(r, "analyst", "")) else str(getattr(r, "analyst", ""))})
    return store


def with_case_columns(alert_df, store):
    out = alert_df.copy()
    out["case_status"] = [case_status(store, u) for u in out.user_id]
    out["analyst_note"] = [latest_note(store, u) for u in out.user_id]
    return out


# ---------------------------------------------------------------- user drill-down
def feature_deviations(features, uid, feature_cols, top_n=10):
    """z-score of each feature for one user against the whole population, largest |z| first."""
    X = features[feature_cols]
    mu, sd = X.mean(), X.std(ddof=0).replace(0, 1.0)
    z = ((features.set_index("user_id").loc[uid, feature_cols] - mu) / sd).astype(float)
    df = pd.DataFrame({"feature": z.index, "z_score": z.values, "value": features.set_index("user_id").loc[uid, feature_cols].values,
                       "population_mean": mu.values})
    return df.reindex(df.z_score.abs().sort_values(ascending=False).index).head(top_n).reset_index(drop=True)


def activity_timeline(file_df, email_df, uid):
    """Daily file-event and e-mail counts for one user (long format for plotting)."""
    f = file_df[file_df.user_id == uid]
    e = email_df[email_df.user_id == uid]
    fd = f.groupby(f.timestamp.dt.normalize()).size().rename("File events")
    ed = e.groupby(e.timestamp.dt.normalize()).size().rename("E-mails sent")
    ex = e[e.external.astype(bool)] if "external" in e else e.iloc[0:0]
    xd = ex.groupby(ex.timestamp.dt.normalize()).size().rename("External e-mails")
    wide = pd.concat([fd, ed, xd], axis=1).fillna(0)
    if wide.empty:
        return pd.DataFrame(columns=["date", "series", "count"])
    full = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
    wide = wide.reindex(full, fill_value=0)
    long = wide.reset_index(names="date").melt(id_vars="date", var_name="series", value_name="count")
    return long


def peer_comparison(features, users, uid, feature_cols, top_n=6):
    """Compare a user with the average of their department on the features where they deviate most."""
    dept = users.set_index("user_id").loc[uid, "department"]
    ids = users[users.department == dept].user_id
    peers = features[features.user_id.isin(ids) & (features.user_id != uid)]
    dev = feature_deviations(features, uid, feature_cols, top_n)
    dev["department_mean"] = [peers[f].mean() if len(peers) else np.nan for f in dev.feature]
    dev["department"] = dept
    return dev[["feature", "value", "department_mean", "population_mean", "z_score", "department"]]


# ---------------------------------------------------------------- threshold simulator
def threshold_sweep(risk, y_true, thresholds=None):
    """Precision / recall / F1 / alert volume for each alert threshold (0-100 risk scale)."""
    risk = np.asarray(risk, float)
    y = np.asarray(y_true).astype(bool)
    thresholds = np.arange(0, 101, 1) if thresholds is None else np.asarray(thresholds)
    rows = []
    P = int(y.sum())
    for t in thresholds:
        flag = risk >= t
        tp = int((flag & y).sum())
        fp = int((flag & ~y).sum())
        fn = P - tp
        prec = tp / (tp + fp) if tp + fp else 1.0
        rec = tp / P if P else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"threshold": float(t), "alerts": tp + fp, "true_positives": tp, "false_positives": fp,
                     "missed_insiders": fn, "precision": prec, "recall": rec, "f1": f1})
    return pd.DataFrame(rows)


def best_threshold(sweep, min_recall=0.0):
    ok = sweep[sweep.recall >= min_recall]
    if ok.empty:
        ok = sweep
    return ok.sort_values(["f1", "threshold"], ascending=[False, False]).iloc[0]


# ---------------------------------------------------------------- multi-model agreement
def model_rankings(score_fn, model_names, user_ids, top_k=15):
    """score_fn(name) -> risk array aligned with user_ids. Returns (rank table, overlap matrix, consensus)."""
    risks = {m: np.asarray(score_fn(m), float) for m in model_names}
    tops = {}
    ranks = pd.DataFrame(index=list(user_ids))
    for m, r in risks.items():
        order = np.argsort(-r)
        tops[m] = set(np.asarray(user_ids)[order[:top_k]])
        rk = np.empty(len(r), int)
        rk[order] = np.arange(1, len(r) + 1)
        ranks[m] = rk
    ov = pd.DataFrame([[len(tops[a] & tops[b]) / max(1, len(tops[a] | tops[b])) for b in model_names] for a in model_names],
                      index=model_names, columns=model_names)
    votes = pd.Series({u: sum(u in tops[m] for m in model_names) for u in user_ids}, name="models_flagging")
    consensus = votes[votes > 0].sort_values(ascending=False).rename_axis("user_id").reset_index()
    consensus["mean_rank"] = [ranks.loc[u].mean() for u in consensus.user_id]
    return ranks, ov, consensus.sort_values(["models_flagging", "mean_rank"], ascending=[False, True]).reset_index(drop=True)


# ---------------------------------------------------------------- export
def alerts_csv(alert_df, store):
    return with_case_columns(alert_df, store).drop(columns=["anomaly_score"], errors="ignore").to_csv(index=False).encode("utf-8")


def alerts_pdf(alert_df, store, model_name, drivers=None, generated=None):
    """One- or two-page PDF incident summary (ReportLab). drivers: {user_id: 'feature (+z), ...'}"""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=14 * mm, rightMargin=14 * mm,
                            topMargin=14 * mm, bottomMargin=14 * mm, title="Insider threat alert report")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=17, textColor=colors.HexColor("#1E3A8A"), alignment=0)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12)
    small = ParagraphStyle("s", parent=body, fontSize=7.5, leading=9.5)
    navy = colors.HexColor("#1E3A8A")
    tbl = with_case_columns(alert_df, store)
    drivers = drivers or {}
    n_high = int((tbl.risk_level == "High").sum())
    n_open = int(tbl.case_status.isin(OPEN_STATUSES).sum())
    when = (generated or datetime.now()).strftime("%Y-%m-%d %H:%M")
    story = [Paragraph("Insider Threat Detection - Alert Report", h1),
             Paragraph(f"Generated {when} &nbsp;|&nbsp; Model: <b>{model_name}</b> &nbsp;|&nbsp; Alerts: <b>{len(tbl)}</b> "
                       f"&nbsp;|&nbsp; High severity: <b>{n_high}</b> &nbsp;|&nbsp; Open cases: <b>{n_open}</b>", body),
             Spacer(1, 5 * mm)]

    def P(t, st=small):
        return Paragraph(str(t).replace("&", "&amp;").replace("<", "&lt;"), st)

    rows = [["Rank", "Entity", "Risk", "Severity", "Case status", "Top drivers", "Analyst note"]]
    for r in tbl.itertuples():
        rows.append([r.rank, r.user_id, f"{r.risk_score:.1f}", r.risk_level, P(r.case_status),
                     P(drivers.get(r.user_id, "")), P(r.analyst_note)])
    t = Table(rows, repeatRows=1, colWidths=[12 * mm, 20 * mm, 16 * mm, 20 * mm, 36 * mm, 90 * mm, 50 * mm])
    style = [("BACKGROUND", (0, 0), (-1, 0), navy), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
             ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8),
             ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C9D2E0")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7FC")])]
    sev = {"High": "#B91C1C", "Medium": "#B45309", "Low": "#15803D"}
    for i, r in enumerate(tbl.itertuples(), start=1):
        style.append(("TEXTCOLOR", (3, i), (3, i), colors.HexColor(sev.get(r.risk_level, "#000000"))))
    t.setStyle(TableStyle(style))
    story += [t, Spacer(1, 5 * mm)]
    log = cases_to_frame(store)
    if not log.empty:
        story.append(Paragraph("Case audit trail", ParagraphStyle("h2", parent=body, fontName="Helvetica-Bold", fontSize=11, textColor=navy)))
        lr = [["Time", "Entity", "Status", "Analyst", "Note"]] + [[r.time, r.user_id, P(r.status), r.analyst, P(r.note)] for r in log.itertuples()]
        lt = Table(lr, repeatRows=1, colWidths=[34 * mm, 20 * mm, 40 * mm, 24 * mm, 126 * mm])
        lt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), navy), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                ("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C9D2E0")),
                                ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(lt)
    story += [Spacer(1, 4 * mm), P("Risk scores are model outputs on an unsupervised anomaly scale and indicate behaviour that deviates from the "
                                   "population, not proof of malicious intent. Demo data is synthetic.")]
    doc.build(story)
    return buf.getvalue()
