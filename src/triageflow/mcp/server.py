"""TriageFlow's own MCP server — the agent's hands.

Tools (at least three that *do* something, per the challenge):
  1. validate_triage_input   — data-quality gate (missing/implausible/stale/contradictory)
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

from ..core import sats_rules as R
from ..core.audit import get_audit
from ..core.lexicon import suggest_codes
from ..core.models import Encounter, VitalSigns
from ..core.scoring import compute_tews, score_encounter
from ..core.validation import validate_vitals

mcp = FastMCP("triageflow")
audit = get_audit()


def _parse_encounter(data: dict[str, Any]) -> Encounter:
    if "vitals" in data and isinstance(data["vitals"], dict):
        data = {**data, "vitals": VitalSigns(**data["vitals"])}
    data.setdefault("encounter_id", "ADHOC")
    return Encounter(**data)


@mcp.tool()
def validate_triage_input(vitals: dict[str, Any], encounter_id: str = "ADHOC") -> dict[str, Any]:
    """Validate measured vitals: missing required params, implausible values, wrong units,
    stale measurements, simple contradictions. Returns a structured report."""
    audit.record("tool_call", encounter_id=encounter_id, tool="validate_triage_input",
                 payload={"vitals": vitals})
    report = validate_vitals(VitalSigns(**vitals))
    out = report.model_dump(mode="json")
    audit.record("tool_result", encounter_id=encounter_id, tool="validate_triage_input",
                 payload=out)
    return out


@mcp.tool()
def score_triage(encounter: dict[str, Any]) -> dict[str, Any]:
    """Compute the deterministic SATS Adult priority for an encounter. Returns the proposed
    priority, TEWS total/band, and a sourced evidence list. This is a PROPOSAL — it is not
    confirmed until request_clinician_approval records a clinician's decision."""
    enc = _parse_encounter(encounter)
    audit.record("tool_call", encounter_id=enc.encounter_id, tool="score_triage",
                 payload={"encounter": encounter})
    result = score_encounter(enc)
    out = result.model_dump(mode="json")
    audit.record("tool_result", encounter_id=enc.encounter_id, tool="score_triage",
                 payload={"proposed_priority": out.get("proposed_priority"),
                          "tews_total": out.get("tews_total"),
                          "complete": out.get("complete"),
                          "n_citations": result.citation_count()})
    return out


@mcp.tool()
def check_deterioration(encounter_id: str, baseline_vitals: dict[str, Any],
                        recheck_vitals: dict[str, Any]) -> dict[str, Any]:
    """Compare a re-check against the baseline. If the TEWS rose or priority escalated,
    raise a SECOND-LOOK flag (sourced). This never changes the queue — it asks a human to
    re-look. Returns {escalated, baseline_tews, recheck_tews, delta, reasons[]}."""
    base_v, re_v = VitalSigns(**baseline_vitals), VitalSigns(**recheck_vitals)
    base_total, _, _ = compute_tews(base_v)
    re_total, re_ev, missing = compute_tews(re_v)
    audit.record("tool_call", encounter_id=encounter_id, tool="check_deterioration",
                 payload={"baseline": baseline_vitals, "recheck": recheck_vitals})

    reasons: list[dict[str, Any]] = []
    escalated = False
    if base_total is not None and re_total is not None:
        delta = re_total - base_total
        if delta >= 2:
            escalated = True
            reasons.append({"reason": f"TEWS rose by {delta} ({base_total}→{re_total})",
                            "source": R.SOURCE_TEWS})
    else:
        delta = None
    # Parameter-level red flags on re-check
    if re_v.spo2 is not None and re_v.spo2 < 92:
        escalated = True
        reasons.append({"reason": f"SpO2 dropped to {re_v.spo2}%",
                        "source": "SATS additional investigations (SpO2 < 92%)"})
    base_band = R.band_for_tews(base_total) if base_total is not None else None
    re_band = R.band_for_tews(re_total) if re_total is not None else None
    if base_band and re_band and R.ACUITY[re_band] > R.ACUITY[base_band]:
        escalated = True
        reasons.append({"reason": f"Band escalated {base_band.value}→{re_band.value}",
                        "source": R.SOURCE_FLOW})

    out = {
        "encounter_id": encounter_id, "escalated": escalated,
        "baseline_tews": base_total, "recheck_tews": re_total, "delta": delta,
        "recheck_band": re_band.value if re_band else None,
        "reasons": reasons, "missing": missing,
        "note": "Second-look flag for a clinician. The agent cannot move the patient.",
    }
    if escalated:
        audit.record("flag", encounter_id=encounter_id, tool="check_deterioration", payload=out)
    audit.record("tool_result", encounter_id=encounter_id, tool="check_deterioration",
                 payload={"escalated": escalated, "delta": delta})
    return out


@mcp.tool()
def request_clinician_approval(encounter_id: str, proposed_priority: str | None,
                               clinician_name: str, decision: str,
                               override_priority: str | None = None,
                               reason: str = "") -> dict[str, Any]:
    """The human gate on the irreversible action. A named clinician APPROVES the proposed
    priority or OVERRIDES it (SHCP discretion). Records the decision to the audit log and
    returns the CONFIRMED priority that may now reach the queue. decision ∈ {approve, override}."""
    if decision not in ("approve", "override"):
        raise ValueError("decision must be 'approve' or 'override'")
    if not clinician_name.strip():
        raise ValueError("clinician_name is required — the gate needs a named approver")
    confirmed = proposed_priority if decision == "approve" else override_priority
    if decision == "override" and not confirmed:
        raise ValueError("override requires override_priority")
    confirmed = str(confirmed) if confirmed is not None else None
    record = {
        "encounter_id": encounter_id, "proposed_priority": proposed_priority,
        "decision": decision, "confirmed_priority": confirmed,
        "clinician": clinician_name, "reason": reason,
    }
    audit.record("approval", encounter_id=encounter_id, actor=clinician_name,
                 tool="request_clinician_approval", payload=record)
    record["confirmed"] = True
    record["source"] = R.SOURCE_FLOW + " (SHCP discretion)"
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


def build_server() -> FastMCP:
    return mcp


if __name__ == "__main__":
    mcp.run()
