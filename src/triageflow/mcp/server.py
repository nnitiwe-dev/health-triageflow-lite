"""TriageFlow's own MCP server — the agent's hands.

Tools (at least three that *do* something, per the challenge):
  1. validate_triage_input   — data-quality gate (missing/impossible/extreme/stale/contradictory)
  2. score_triage            — deterministic SATS priority + sourced evidence  [ACTION]
  3. check_deterioration     — compare a re-check to baseline, raise a 2nd-look flag  [ACTION]
  4. request_clinician_approval — the human-in-the-loop gate on the irreversible action  [ACTION]
  5. suggest_discriminators  — deterministic keyword map from free text -> candidate SATS
                               discriminator codes, for the nurse to CONFIRM (never auto-applied)

Every call is written to the append-only audit log (inputs, outputs, timestamp, errors).
The LLM orchestrates these tools; it never computes a priority itself.

Run as a stdio MCP server:  python -m triageflow.mcp.server
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import ValidationError

from ..core import sats_rules as R
from ..core.audit import get_audit
from ..core.config import settings
from ..core.explain import param_label, param_label_lower
from ..core.lexicon import suggest_codes
from ..core.models import Encounter, VitalSigns
from ..core.scoring import (
    additional_investigation_flags,
    compute_tews,
    derived_signs,
    score_encounter,
    vitals_priority,
)
from ..core.validation import validate_vitals

mcp = FastMCP("triageflow")
audit = get_audit()


def _parse_encounter(data: dict[str, Any]) -> Encounter:
    if "vitals" in data and isinstance(data["vitals"], dict):
        data = {**data, "vitals": VitalSigns(**data["vitals"])}
    data.setdefault("encounter_id", "ADHOC")
    return Encounter(**data)


def _vitals_or_issues(vitals: dict[str, Any]) -> tuple[VitalSigns | None, list[dict[str, Any]]]:
    """Parse vitals; a value the schema rejects (e.g. pain 14) becomes a readable issue for
    the nurse instead of a crashed tool call."""
    try:
        return VitalSigns(**vitals), []
    except ValidationError as e:
        issues = []
        for err in e.errors():
            field = str(err["loc"][0]) if err.get("loc") else "vitals"
            issues.append({"field": field, "severity": "error",
                           "message": f"{param_label(field)}: {err.get('msg', 'invalid value')} "
                                      f"(entered: {err.get('input')!r})."})
        return None, issues


@mcp.tool()
def validate_triage_input(vitals: dict[str, Any], encounter_id: str = "ADHOC",
                          age_years: int | None = None) -> dict[str, Any]:
    """Validate measured vitals: missing required params, impossible values (errors), extreme
    but possible values (warnings — still scored), wrong units, stale measurements, simple
    contradictions, and the adult-pathway age limit. Returns a structured report."""
    audit.record("tool_call", encounter_id=encounter_id, tool="validate_triage_input",
                 payload={"vitals": vitals, "age_years": age_years})
    parsed, issues = _vitals_or_issues(vitals)
    if parsed is None:
        out = {"ok": False, "issues": issues, "missing_required": []}
    else:
        out = validate_vitals(parsed, age_years=age_years).model_dump(mode="json")
    audit.record("tool_result", encounter_id=encounter_id, tool="validate_triage_input",
                 payload=out)
    return out


@mcp.tool()
def score_triage(encounter: dict[str, Any]) -> dict[str, Any]:
    """Compute the deterministic SATS Adult priority for an encounter. Returns the proposed
    priority, TEWS total/band, a sourced evidence list and the SATS 'additional investigation'
    actions. This is a PROPOSAL — it is not confirmed until request_clinician_approval records
    a clinician's decision."""
    enc = _parse_encounter(encounter)
    audit.record("tool_call", encounter_id=enc.encounter_id, tool="score_triage",
                 payload={"encounter": encounter})
    result = score_encounter(enc)
    out = result.model_dump(mode="json")
    audit.record("tool_result", encounter_id=enc.encounter_id, tool="score_triage",
                 payload={"proposed_priority": out.get("proposed_priority"),
                          "tews_total": out.get("tews_total"),
                          "complete": out.get("complete"),
                          "actions": [a.code for a in result.actions],
                          "n_citations": result.citation_count()})
    return out


@mcp.tool()
def check_deterioration(encounter_id: str, baseline_vitals: dict[str, Any],
                        recheck_vitals: dict[str, Any]) -> dict[str, Any]:
    """Compare a re-check against the baseline and raise a SECOND-LOOK flag if the patient
    scores worse. This is a TriageFlow watch convention (not a SATS rule) and never changes
    the queue — it asks a human to re-look.

    It never fails silently: if the re-check is missing a vital, the parameters that WERE
    measured are still compared, and the gap is reported in ``missing`` / ``incomplete``.
    Returns {escalated, baseline_tews, recheck_tews, delta, reasons[], missing[], incomplete}."""
    base_v, re_v = VitalSigns(**baseline_vitals), VitalSigns(**recheck_vitals)
    base_total, base_ev, _ = compute_tews(base_v)
    re_total, re_ev, missing = compute_tews(re_v)
    audit.record("tool_call", encounter_id=encounter_id, tool="check_deterioration",
                 payload={"baseline": baseline_vitals, "recheck": recheck_vitals})

    reasons: list[dict[str, Any]] = []
    watch = R.SOURCE_LOCAL_WATCH

    # 1. Like-for-like TEWS comparison over the parameters present in BOTH sets.
    base_pts = {e.parameter: e for e in base_ev}
    re_pts = {e.parameter: e for e in re_ev}
    common = [p for p in re_pts if p in base_pts]
    delta = sum((re_pts[p].points or 0) - (base_pts[p].points or 0) for p in common)
    worse = [p for p in common if (re_pts[p].points or 0) > (base_pts[p].points or 0)]
    if delta >= 2:
        scope = (f"{base_total}→{re_total}" if base_total is not None and re_total is not None
                 else f"on the {len(common)} vitals measured both times")
        reasons.append({"reason": f"TEWS rose by {delta} ({scope}): "
                                  + ", ".join(param_label_lower(p) for p in worse),
                        "source": watch})

    # 2. The priority the vitals alone imply (TEWS band, alertness, glucose, pain, SpO2).
    base_pr, re_pr = vitals_priority(base_v), vitals_priority(re_v)
    if re_pr is not None and (base_pr is None or R.ACUITY[re_pr] > R.ACUITY[base_pr]):
        reasons.append({"reason": f"Vitals now indicate {re_pr.value} "
                                  f"(was {base_pr.value if base_pr else 'not scored'})",
                        "source": watch})

    # 3. New red-flag findings on the re-check, named individually.
    def _new(flags_then, flags_now):
        then = {e.parameter for e in flags_then}
        return [e for e in flags_now if e.parameter not in then]
    for e in _new(derived_signs(base_v) + additional_investigation_flags(base_v),
                  derived_signs(re_v) + additional_investigation_flags(re_v)):
        label = e.observed if e.kind == "discriminator" else f"{e.band_label} — {e.observed}"
        reasons.append({"reason": f"New finding: {label}", "source": e.source})

    escalated = bool(reasons)
    base_band = R.band_for_tews(base_total) if base_total is not None else None
    re_band = R.band_for_tews(re_total) if re_total is not None else None
    out = {
        "encounter_id": encounter_id, "escalated": escalated,
        "baseline_tews": base_total, "recheck_tews": re_total,
        "delta": delta if common else None,
        "recheck_band": re_band.value if re_band else None,
        "baseline_band": base_band.value if base_band else None,
        "recheck_priority": re_pr.value if re_pr else None,
        "reasons": reasons, "missing": missing, "incomplete": bool(missing),
        "note": ("Second-look flag for a clinician. The agent cannot move the patient."
                 + (" Re-check is incomplete — missing: "
                    + ", ".join(param_label_lower(m) for m in missing) + "." if missing else "")),
    }
    if escalated:
        audit.record("flag", encounter_id=encounter_id, tool="check_deterioration", payload=out)
    audit.record("tool_result", encounter_id=encounter_id, tool="check_deterioration",
                 payload={"escalated": escalated, "delta": out["delta"], "missing": missing})
    return out


_VALID_PRIORITIES = {p.value for p in R.Priority}


@mcp.tool()
def request_clinician_approval(encounter_id: str, proposed_priority: str | None,
                               clinician_name: str, decision: str,
                               override_priority: str | None = None,
                               reason: str = "") -> dict[str, Any]:
    """The human gate on the irreversible action. A named clinician APPROVES the proposed
    priority or OVERRIDES it (senior clinician's discretion). Records the decision to the
    audit log and returns the CONFIRMED priority that may now reach the queue.

    Enforced here, on the server (never only in a UI):
      * a named approver is required;
      * 'approve' needs a proposed priority — an incomplete assessment cannot be approved;
      * 'override' needs a valid priority AND a written reason."""
    if decision not in ("approve", "override"):
        raise ValueError("decision must be 'approve' or 'override'")
    if not (clinician_name or "").strip():
        raise ValueError("clinician_name is required — the gate needs a named approver")
    if decision == "approve":
        if not proposed_priority:
            raise ValueError("an incomplete assessment cannot be approved — complete the "
                             "vitals, or override with a priority and a reason")
        confirmed = str(proposed_priority)
    else:
        if override_priority not in _VALID_PRIORITIES:
            raise ValueError("override requires a valid override_priority "
                             f"({', '.join(sorted(_VALID_PRIORITIES))})")
        if not (reason or "").strip():
            raise ValueError("override requires a written reason")
        confirmed = str(override_priority)
    record = {
        "encounter_id": encounter_id, "proposed_priority": proposed_priority,
        "decision": decision, "confirmed_priority": confirmed,
        "clinician": clinician_name.strip(), "reason": (reason or "").strip(),
    }
    audit.record("approval", encounter_id=encounter_id, actor=clinician_name.strip(),
                 tool="request_clinician_approval", payload=record)
    record["confirmed"] = True
    record["source"] = R.SOURCE_FLOW + " (senior healthcare professional's discretion)"
    return record


@mcp.tool()
def suggest_discriminators(text: str) -> dict[str, Any]:
    """Deterministic keyword map from a free-text complaint to CANDIDATE SATS discriminator
    codes. These are suggestions for the nurse to confirm — never auto-applied to a score."""
    hits = []
    for code in suggest_codes(text):
        d = R.DISCRIMINATORS_BY_CODE[code]
        hits.append({"code": code, "label": d.label, "implies": d.level.value,
                     "source": d.source})
    return {"text": text, "candidates": hits,
            "note": "Candidates only — the nurse confirms before these affect any priority."}


@mcp.tool()
def fetch_fhir_vitals(patient_id: str) -> dict[str, Any]:
    """Import a patient's vitals from the configured FHIR R4 server (EMR ingest), so the nurse
    does not re-type what the EMR already holds. Returns {vitals, source} or a graceful error.
    Disabled unless TRIAGEFLOW_FHIR_BASE_URL is set; the result is still nurse-confirmed."""
    from ..agent import fhir_emr as F
    if not F.fhir_enabled():
        return {"ok": False, "error": "FHIR integration is off (set TRIAGEFLOW_FHIR_BASE_URL)."}
    audit.record("tool_call", encounter_id=patient_id, tool="fetch_fhir_vitals",
                 payload={"patient_id": patient_id})
    try:
        vitals = F.import_patient_vitals(patient_id)
    except F.FHIRUnavailable as e:
        audit.record("tool_error", encounter_id=patient_id, tool="fetch_fhir_vitals",
                     payload={"error": str(e)})
        return {"ok": False, "error": str(e)}
    out = {"ok": True, "vitals": vitals.model_dump(mode="json"),
           "source": f"FHIR Observation (patient {patient_id}) @ {settings.fhir_base_url}"}
    audit.record("tool_result", encounter_id=patient_id, tool="fetch_fhir_vitals",
                 payload={"source": out["source"]})
    return out


def build_server() -> FastMCP:
    return mcp


if __name__ == "__main__":
    mcp.run()
