"""Deterministic SATS Adult scoring engine.

This is pure, side-effect-free Python. The LLM never computes a priority — it only
orchestrates and explains. That is the defensibility guarantee: given the same vitals,
this returns the same score every time, and every point cites its source.
"""

from __future__ import annotations

from . import sats_rules as R
from .models import Encounter, Evidence, TriageResult, VitalSigns

# The 7 TEWS parameters we require to call a TEWS "complete".
REQUIRED_TEWS_PARAMS = [
    "respiratory_rate",
    "heart_rate",
    "systolic_bp",
    "temperature_c",
    "avpu",
    "mobility",
    "trauma",
]


def _band_points(value: float, bands: list[R.Band]) -> R.Band | None:
    for b in bands:
        if b.test(value):
            return b
    return None


def score_vital(param: str, vitals: VitalSigns) -> Evidence | None:
    """Return the TEWS Evidence for one parameter, or None if the value is absent."""
    v = getattr(vitals, param)
    if v is None:
        return None

    if param == "respiratory_rate":
        b = _band_points(v, R.RR_BANDS)
    elif param == "heart_rate":
        b = _band_points(v, R.HR_BANDS)
    elif param == "systolic_bp":
        b = _band_points(v, R.SBP_BANDS)
    elif param == "temperature_c":
        b = _band_points(v, R.TEMP_BANDS)
    elif param == "mobility":
        pts, label = R.MOBILITY_POINTS[v]
        return Evidence(kind="tews", parameter=param, observed=v.value,
                        points=pts, band_label=label, source=R.SOURCE_TEWS)
    elif param == "avpu":
        pts, label = R.AVPU_POINTS[v]
        return Evidence(kind="tews", parameter=param, observed=v.value,
                        points=pts, band_label=label, source=R.SOURCE_TEWS)
    elif param == "trauma":
        pts, label = R.TRAUMA_POINTS[bool(v)]
        return Evidence(kind="tews", parameter=param, observed=str(v),
                        points=pts, band_label=label, source=R.SOURCE_TEWS)
    else:
        return None

    if b is None:
        return None
    return Evidence(kind="tews", parameter=param, observed=str(v),
                    points=b.points, band_label=b.label, source=R.SOURCE_TEWS)


def compute_tews(vitals: VitalSigns) -> tuple[int | None, list[Evidence], list[str]]:
    """Return (total, per-parameter evidence, missing params).

    Total is None if any required parameter is missing (we never guess a TEWS).
    """
    evidence: list[Evidence] = []
    missing: list[str] = []
    for param in REQUIRED_TEWS_PARAMS:
        ev = score_vital(param, vitals)
        if ev is None:
            missing.append(param)
        else:
            evidence.append(ev)

    if missing:
        return None, evidence, missing
    total = sum(e.points or 0 for e in evidence)
    return total, evidence, missing


def discriminator_evidence(codes: list[str]) -> tuple[R.Priority | None, list[Evidence]]:
    """Resolve asserted discriminator codes to their highest implied priority + evidence."""
    evidence: list[Evidence] = []
    level: R.Priority | None = None
    for code in codes:
        d = R.DISCRIMINATORS_BY_CODE.get(code)
        if d is None:
            continue
        evidence.append(Evidence(
            kind="discriminator", parameter=d.code, observed=d.label,
            implies=d.level, source=d.source,
        ))
        if level is None or R.ACUITY[d.level] > R.ACUITY[level]:
            level = d.level
    return level, evidence


def additional_investigation_flags(vitals: VitalSigns) -> list[Evidence]:
    """SpO2 / glucose flags that can raise priority. Sourced. Not diagnoses."""
    flags: list[Evidence] = []
    if vitals.spo2 is not None and vitals.spo2 < 92:
        flags.append(Evidence(
            kind="additional_investigation", parameter="spo2",
            observed=f"{vitals.spo2}%", implies=R.Priority.ORANGE,
            source="SATS Training Manual 2012, additional investigations (SpO2 < 92%)",
        ))
    if vitals.glucose_mmol is not None and vitals.glucose_mmol < 3:
        flags.append(Evidence(
            kind="additional_investigation", parameter="glucose_mmol",
            observed=f"{vitals.glucose_mmol} mmol/L", implies=R.Priority.RED,
            source="SATS Training Manual 2012, hypoglycaemia (glucose < 3 mmol/L) = emergency",
        ))
    # Pain score (0-10) maps to the SATS pain discriminator. The nurse enters this directly
    # (a structured assessment, not a free-text guess), so it contributes with its source.
    if vitals.pain_score is not None and vitals.pain_score >= 4:
        code = "severe_pain" if vitals.pain_score >= 7 else "moderate_pain"
        d = R.DISCRIMINATORS_BY_CODE[code]
        flags.append(Evidence(
            kind="discriminator", parameter=code,
            observed=f"{d.label} ({vitals.pain_score}/10)", band_label=d.label,
            implies=d.level, source=d.source,
        ))
    return flags


def _max_priority(*levels: R.Priority | None) -> R.Priority | None:
    present = [lv for lv in levels if lv is not None]
    if not present:
        return None
    return max(present, key=lambda lv: R.ACUITY[lv])


def score_encounter(enc: Encounter) -> TriageResult:
    """Full deterministic SATS scoring for an encounter.

    Final priority = highest acuity among: TEWS band, discriminator level, and any
    additional-investigation flag. If required vitals are missing AND no discriminator
    already forces a level, the result is marked incomplete (no band shown).
    """
    result = TriageResult(encounter_id=enc.encounter_id, ruleset_version=R.RULESET_VERSION)

    tews_total, tews_ev, missing = compute_tews(enc.vitals)
    disc_level, disc_ev = discriminator_evidence(enc.discriminator_codes)
    inv_ev = additional_investigation_flags(enc.vitals)
    inv_level = _max_priority(*[e.implies for e in inv_ev])

    result.evidence = tews_ev + disc_ev + inv_ev
    result.missing = missing

    tews_band = R.band_for_tews(tews_total) if tews_total is not None else None
    result.tews_total = tews_total
    result.tews_band = tews_band
    result.discriminator_level = disc_level

    proposed = _max_priority(tews_band, disc_level, inv_level)

    # Completeness: we can propose a defensible band if TEWS is complete, OR a
    # discriminator/flag already forces a level regardless of missing vitals.
    forced = _max_priority(disc_level, inv_level)
    if tews_total is None and forced is None:
        result.complete = False
        result.warnings.append(
            "Incomplete: required vitals missing and no overriding sign — "
            "cannot compute a defensible TEWS. Needs: " + ", ".join(missing)
        )
        result.proposed_priority = None
        return result

    result.complete = True
    result.proposed_priority = proposed
    if proposed is not None:
        result.target_time = R.TARGET_TIME[proposed]
        result.disposition = R.DISPOSITION[proposed]

    # If a sign forced a level but TEWS was incomplete, note it stayed defensible.
    if tews_total is None and forced is not None:
        result.warnings.append(
            "Priority set by a clinical sign; TEWS incomplete (missing: "
            + ", ".join(missing) + "). Recommend completing vitals for re-score."
        )
    return result
