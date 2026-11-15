"""Turn a TriageResult into plain language a nurse, a doctor and an auditor can all read.

Deterministic formatting only — no model. The same result always produces the same words.
The structure answers, in order, the questions people actually ask at the desk:

    1. What priority, and how soon must this patient be seen?   (headline / target / where)
    2. Why?                                                      (decided_by / reasons)
    3. What do I do now?                                         (actions)
    4. Anything I should double-check?                           (cautions)
    5. Show me the working.                                      (tews table + sources)
"""

from __future__ import annotations

from typing import Any

from . import sats_rules as R

# parameter -> (label, unit)
PARAM_LABEL: dict[str, tuple[str, str]] = {
    "respiratory_rate": ("Breathing rate", "/min"),
    "heart_rate": ("Pulse rate", "bpm"),
    "systolic_bp": ("Systolic BP", "mmHg"),
    "temperature_c": ("Temperature", "°C"),
    "avpu": ("Alertness (AVPU)", ""),
    "mobility": ("Mobility", ""),
    "trauma": ("Trauma", ""),
    "spo2": ("SpO2", "%"),
    "glucose_mmol": ("Glucose", "mmol/L"),
    "pain_score": ("Pain score", "/10"),
}

TARGET_TEXT: dict[str, str] = {
    "RED": "See immediately — take to resus",
    "ORANGE": "See within 10 minutes",
    "YELLOW": "See within 1 hour",
    "GREEN": "See within 4 hours",
    "BLUE": "Refer for certification within 2 hours",
}

_LEVEL_WORD = {"RED": "emergency sign", "ORANGE": "very urgent sign", "YELLOW": "urgent sign"}


def param_label(param: str) -> str:
    return PARAM_LABEL.get(param, (param.replace("_", " "), ""))[0]


def param_label_lower(param: str) -> str:
    """Label for the middle of a sentence: 'systolic BP', 'alertness (AVPU)', 'SpO2'."""
    label = param_label(param)
    return label if label.startswith("SpO2") else label[:1].lower() + label[1:]


def _acuity(priority: str | None) -> str:
    try:
        return R.ACUITY_NAME[R.Priority(priority)] if priority else ""
    except ValueError:
        return ""


def _tews_value(e: dict[str, Any]) -> str:
    p, obs = e.get("parameter"), str(e.get("observed", ""))
    if p in ("avpu", "mobility"):
        return e.get("band_label") or obs
    if p == "trauma":
        return "Yes" if obs.lower() == "true" else "No"
    unit = PARAM_LABEL.get(p, ("", ""))[1]
    return f"{obs} {unit}".strip()


def _tews_plain(e: dict[str, Any]) -> str:
    """A short, human description of an abnormal TEWS finding, e.g. 'Fast breathing — 28 /min'."""
    p, value = e.get("parameter"), _tews_value(e)
    try:
        num = float(e.get("observed"))
    except (TypeError, ValueError):
        num = None
    if p == "respiratory_rate" and num is not None:
        return f"{'Slow' if num < 9 else 'Fast'} breathing — {value}"
    if p == "heart_rate" and num is not None:
        return f"{'Slow' if num < 51 else 'Fast'} pulse — {value}"
    if p == "systolic_bp" and num is not None:
        return f"{'Low' if num < 101 else 'Very high'} blood pressure — systolic {value}"
    if p == "temperature_c" and num is not None:
        return f"{'Low temperature' if num < 35 else 'High temperature (fever)'} — {value}"
    if p == "avpu":
        return f"Not fully alert — {value.lower()}"
    if p == "mobility":
        return ("Needs help to walk" if "help" in value.lower()
                else "On a stretcher / unable to walk")
    if p == "trauma":
        return "Trauma-related presentation"
    return f"{param_label(p)} {value}"


def _sign_text(e: dict[str, Any]) -> str:
    if e.get("kind") == "additional_investigation":
        return f"{e.get('band_label')} — {e.get('observed')}"
    if e.get("kind") == "safety_rule":
        return f"Low oxygen saturation — SpO2 {e.get('observed')}"
    return str(e.get("observed") or e.get("band_label") or e.get("parameter"))


def build_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Structured, plain-language view of a scored result (see module docstring)."""
    priority = result.get("proposed_priority")
    evidence = result.get("evidence") or []
    tews_rows = [e for e in evidence if e.get("kind") == "tews"]
    signs = [e for e in evidence if e.get("kind") != "tews"]
    total, band = result.get("tews_total"), result.get("tews_band")

    reasons: list[dict[str, Any]] = []
    for e in signs:
        level = e.get("implies")
        local = e.get("kind") == "safety_rule"
        reasons.append({
            "text": _sign_text(e),
            "detail": ("local safety rule, not SATS" if local
                       else _LEVEL_WORD.get(level, "sign")),
            "level": level, "kind": "safety" if local else "sign",
            "deciding": bool(priority) and level == priority,
        })
    for e in tews_rows:
        if (e.get("points") or 0) > 0:
            reasons.append({
                "text": _tews_plain(e), "detail": f"+{e.get('points')} TEWS",
                "level": None, "kind": "tews",
                "deciding": bool(priority) and band == priority,
            })

    decided_by = [r["text"].split(" — ")[0] for r in reasons if r["kind"] != "tews" and r["deciding"]]
    if priority and band == priority and total is not None:
        decided_by.append(f"TEWS {total}")

    tews = {
        "total": total, "band": band, "band_name": _acuity(band), "complete": total is not None,
        "rows": [{"parameter": e.get("parameter"), "label": param_label(e.get("parameter")),
                  "value": _tews_value(e), "points": e.get("points") or 0,
                  "range": e.get("band_label")} for e in tews_rows],
        "missing": [param_label(m) for m in result.get("missing") or []],
    }

    if result.get("out_of_scope"):
        status = "out_of_scope"
        headline = "Adult SATS does not apply"
        sentence = (result.get("warnings") or ["Paediatric patient — use the paediatric chart."])[0]
    elif not result.get("complete") or not priority:
        status = "incomplete"
        headline = "No priority yet"
        need = ", ".join(param_label_lower(m) for m in result.get("missing") or []) \
            or "more information"
        sentence = f"No priority can be given yet. Still needed: {need}."
    else:
        status = "ready"
        headline = f"{priority} — {_acuity(priority)}"
        if decided_by:
            why = "Decided by: " + "; ".join(decided_by) + "."
        else:
            why = "No urgent signs found."
        if total is None:
            tews_txt = "" if priority == "RED" else "TEWS not complete yet."
        elif band == priority:
            tews_txt = ""
        else:
            tews_txt = f"TEWS {total} (on its own: {str(_acuity(band)).lower()})."
        if priority == "GREEN" and total is not None and not decided_by[:-1]:
            why = f"No urgent signs and vitals in the routine range (TEWS {total})."
            tews_txt = ""
        sentence = " ".join(x for x in (
            f"{priority} ({_acuity(priority).lower()}): {TARGET_TEXT.get(priority, '').lower()}.",
            why, tews_txt) if x)

    return {
        "status": status,
        "priority": priority if status == "ready" else None,
        "acuity": _acuity(priority) if status == "ready" else "",
        "headline": headline,
        "target": TARGET_TEXT.get(priority, "") if status == "ready" else "",
        "where": (result.get("disposition") or "") if status == "ready" else "",
        "decided_by": decided_by,
        "reasons": reasons,
        "tews": tews,
        "actions": [{"text": a.get("text"), "reason": a.get("reason"), "source": a.get("source")}
                    for a in result.get("actions") or []],
        "cautions": list(result.get("warnings") or []),
        "sentence": sentence,
    }


def format_text(summary: dict[str, Any]) -> str:
    """Terminal / handover rendering of a summary (used by the CLI demo and the audit log)."""
    lines = [summary["headline"]]
    if summary.get("target"):
        lines.append(f"  {summary['target']} · {summary.get('where', '')}".rstrip(" ·"))
    if summary.get("reasons"):
        lines.append("  Why:")
        for r in summary["reasons"]:
            mark = "*" if r["deciding"] else "-"
            lines.append(f"    {mark} {r['text']} ({r['detail']})")
    t = summary["tews"]
    if t["complete"]:
        lines.append(f"  TEWS {t['total']} ({t['band_name'].lower()} range)")
    elif t["missing"]:
        lines.append("  TEWS incomplete — still needed: " + ", ".join(t["missing"]))
    if summary.get("actions"):
        lines.append("  Do now:")
        lines += [f"    > {a['text']} ({a['reason']})" for a in summary["actions"]]
    if summary.get("cautions"):
        lines.append("  Check:")
        lines += [f"    ! {c}" for c in summary["cautions"]]
    return "\n".join(lines)
