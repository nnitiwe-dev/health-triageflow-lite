"""Deterministic SATS Adult scoring engine.

This is pure, side-effect-free Python. The LLM never computes a priority — it only
orchestrates and explains. That is the defensibility guarantee: given the same vitals,
this returns the same score every time, and every point cites its source.

Anything that is NOT in the SATS adult chart (the SpO2 safety floor, the numeric pain
cut-offs) is labelled as a local convention in its own source string — never passed off
as SATS.
"""

from __future__ import annotations

import math
import re

from . import sats_rules as R
from .explain import param_label_lower
from .models import Action, Encounter, Evidence, TriageResult, VitalSigns

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

_DIABETES = re.compile(r"\b(diabet\w*|dm ?[12]?|t[12]dm|iddm|niddm)\b", re.IGNORECASE)


def fmt_num(v: float) -> str:
    """36.7 -> '36.7', 28.0 -> '28' (what a nurse would write)."""
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def _band_points(value: float, bands: list[R.Band]) -> R.Band | None:
    """Find the TEWS band for a reading.

    The SATS chart prints integer ranges (e.g. RR 9-14, 15-20). A non-integer reading that
    falls *between* two printed ranges (14.5, from a monitor average or an EMR import) is
    scored at the more severe of the two adjacent bands — conservative, and never "missing".
    """
    for b in bands:
        if b.test(value):
            return b
    neighbours = [b for v in (math.floor(value), math.ceil(value)) for b in bands if b.test(v)]
    return max(neighbours, key=lambda b: b.points) if neighbours else None


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
    return Evidence(kind="tews", parameter=param, observed=fmt_num(v),
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


def discriminator_evidence(
        codes: list[str]) -> tuple[R.Priority | None, list[Evidence], list[str]]:
    """Resolve asserted discriminator codes to (highest implied priority, evidence, unknown).

    Unknown codes are returned (never silently dropped) so the caller can warn about them.
    """
    evidence: list[Evidence] = []
    unknown: list[str] = []
    level: R.Priority | None = None
    for code in dict.fromkeys(codes):
        d = R.DISCRIMINATORS_BY_CODE.get(code)
        if d is None:
            unknown.append(code)
            continue
        evidence.append(Evidence(
            kind="discriminator", parameter=d.code, observed=d.label, band_label=d.label,
            implies=d.level, source=d.source,
        ))
        if level is None or R.ACUITY[d.level] > R.ACUITY[level]:
            level = d.level
    return level, evidence, unknown


def derived_signs(vitals: VitalSigns) -> list[Evidence]:
    """Discriminators that follow directly from a structured observation the nurse entered.

    SATS: "Level of consciousness reduced / confused — any patient that is not fully alert"
    is a VERY URGENT sign. If AVPU is anything but Alert, that sign is present by definition;
    it must not depend on a free-text match.
    """
    out: list[Evidence] = []
    if vitals.avpu is not None and vitals.avpu != R.AVPU.ALERT:
        d = R.DISCRIMINATORS_BY_CODE["loc_reduced"]
        _, avpu_label = R.AVPU_POINTS[vitals.avpu]
        out.append(Evidence(
            kind="discriminator", parameter=d.code,
            observed=f"{d.label} (AVPU: {avpu_label.lower()})", band_label=d.label,
            implies=d.level, source=R.SOURCE_LOC,
        ))
    return out


def additional_investigation_flags(vitals: VitalSigns) -> list[Evidence]:
    """Glucose / pain / SpO2 findings that raise priority. Each is sourced. Not diagnoses."""
    flags: list[Evidence] = []
    if vitals.glucose_mmol is not None and vitals.glucose_mmol <= R.HYPOGLYCAEMIA_MAX_MMOL:
        flags.append(Evidence(
            kind="additional_investigation", parameter="glucose_mmol",
            observed=f"{fmt_num(vitals.glucose_mmol)} mmol/L",
            band_label="Hypoglycaemia (glucose 3 mmol/L or less)",
            implies=R.Priority.RED, source=R.SOURCE_HYPO,
        ))
    # Pain score (0-10) maps to the SATS pain discriminator. The nurse enters the number; the
    # cut-offs are a stated convention (see SOURCE_PAIN), and the nurse confirms the result.
    if vitals.pain_score is not None and vitals.pain_score >= R.PAIN_MODERATE_MIN:
        code = "severe_pain" if vitals.pain_score >= R.PAIN_SEVERE_MIN else "moderate_pain"
        d = R.DISCRIMINATORS_BY_CODE[code]
        flags.append(Evidence(
            kind="discriminator", parameter=code,
            observed=f"{d.label} (pain {vitals.pain_score}/10)", band_label=d.label,
            implies=d.level, source=R.SOURCE_PAIN,
        ))
    # Local safety floor — NOT a SATS adult discriminator, and labelled as such.
    if vitals.spo2 is not None and vitals.spo2 < R.SPO2_SAFETY_FLOOR:
        flags.append(Evidence(
            kind="safety_rule", parameter="spo2",
            observed=f"{fmt_num(vitals.spo2)}%",
            band_label=f"SpO2 below {R.SPO2_SAFETY_FLOOR}% (local safety rule, not SATS)",
            implies=R.Priority.ORANGE, source=R.SOURCE_LOCAL_SPO2,
        ))
    return flags


def additional_actions(enc: Encounter, tews_ev: list[Evidence],
                       sign_codes: set[str]) -> list[Action]:
    """The SATS adult chart's "check for additional investigations" prompts for this patient.

    These are the chart's own tasks (measure SpO2, finger-prick glucose, ketone dipstick,
    ECG, pregnancy test) — surfaced so they are not forgotten. The agent does none of them.
    """
    v = enc.vitals
    src = R.SOURCE_INVESTIGATIONS
    actions: list[Action] = []

    rr = next((e for e in tews_ev if e.parameter == "respiratory_rate"), None)
    if v.spo2 is None and rr is not None and (rr.points or 0) >= 1:
        actions.append(Action(
            code="check_spo2",
            text="Measure SpO2 (oxygen saturation) and hand over to the senior clinician for oxygen.",
            reason=f"Breathing rate {rr.observed}/min scores {rr.points} on the TEWS.", source=src))
    if v.spo2 is not None and v.spo2 < R.SPO2_SAFETY_FLOOR:
        actions.append(Action(
            code="oxygen_handover",
            text="Hand over to the senior clinician for oxygen now.",
            reason=f"SpO2 is {fmt_num(v.spo2)}% (below {R.SPO2_SAFETY_FLOOR}%).", source=src))

    if v.glucose_mmol is None:
        why = []
        if v.avpu is not None and v.avpu != R.AVPU.ALERT:
            why.append("the patient is not fully alert")
        if v.mobility == R.Mobility.IMMOBILE:
            why.append("the patient is unable to sit up / needs to lie down")
        if sign_codes & {"seizure_active", "post_ictal"}:
            why.append("seizure")
        text = f"{enc.presenting_complaint} {enc.free_text_notes}"
        if _DIABETES.search(text) or sign_codes & {"diabetic_gt11_ketonuria", "diabetic_gt17"}:
            why.append("history of diabetes")
        if why:
            actions.append(Action(
                code="check_glucose",
                text="Do a finger-prick glucose test and hand over to the senior clinician.",
                reason="Because " + ", and ".join(why) + ".", source=src))
    elif v.glucose_mmol <= R.HYPOGLYCAEMIA_MAX_MMOL:
        actions.append(Action(
            code="hypoglycaemia_resus",
            text="Move to resus and hand over to the senior clinician.",
            reason=f"Glucose is {fmt_num(v.glucose_mmol)} mmol/L (3 or less).", source=src))
    elif v.glucose_mmol >= R.HYPERGLYCAEMIA_MIN_MMOL \
            and not sign_codes & {"diabetic_gt11_ketonuria", "diabetic_gt17"}:
        actions.append(Action(
            code="check_ketones",
            text=("Do a urine dipstick for ketones. If the patient is diabetic, tick the matching "
                  "sign: ketones present = very urgent; glucose over 17 without ketones = urgent."),
            reason=f"Glucose is {fmt_num(v.glucose_mmol)} mmol/L (11 or more).", source=src))

    if "chest_pain" in sign_codes:
        actions.append(Action(
            code="ecg", text="Immediate ECG and hand over to the senior clinician.",
            reason="Chest pain.", source=src))
    if sign_codes & {"abdo_pain", "pregnancy_abdo"} and (enc.sex or "").lower() != "male":
        who = "female patient" if (enc.sex or "").lower() == "female" else "patient (if female)"
        actions.append(Action(
            code="pregnancy_test", text="Urine dipstick and urine pregnancy test.",
            reason=f"Abdominal pain in a {who}.", source=src))
    return actions


def _max_priority(*levels: R.Priority | None) -> R.Priority | None:
    present = [lv for lv in levels if lv is not None]
    if not present:
        return None
    return max(present, key=lambda lv: R.ACUITY[lv])


def vitals_priority(vitals: VitalSigns) -> R.Priority | None:
    """Highest priority implied by the vitals alone (TEWS band if complete, plus any derived
    sign / flag). Used by the deterioration watch to compare a re-check with its baseline."""
    total, _, _ = compute_tews(vitals)
    band = R.band_for_tews(total) if total is not None else None
    flags = derived_signs(vitals) + additional_investigation_flags(vitals)
    return _max_priority(band, *[e.implies for e in flags])


def score_encounter(enc: Encounter) -> TriageResult:
    """Full deterministic SATS scoring for an encounter.

    Final priority = highest acuity among: TEWS band, confirmed + derived discriminators, and
    any additional-investigation / safety flag. If required vitals are missing AND no sign
    already forces a level, the result is marked incomplete (no band shown).
    """
    result = TriageResult(encounter_id=enc.encounter_id, ruleset_version=R.RULESET_VERSION)

    # Scope guard: the adult chart must not be applied to a child.
    if enc.age_years is not None and enc.age_years < R.ADULT_MIN_AGE_YEARS:
        result.out_of_scope = True
        result.complete = False
        result.warnings.append(
            f"Patient is {enc.age_years} years old — the ADULT SATS chart does not apply. "
            "Use the paediatric SATS chart (emergency signs still go straight to resus).")
        return result

    tews_total, tews_ev, missing = compute_tews(enc.vitals)
    disc_level, disc_ev, unknown = discriminator_evidence(enc.discriminator_codes)

    # Derived signs + flags; drop any the nurse already confirmed (no duplicate rows).
    seen = {e.parameter for e in disc_ev}
    auto_ev = [e for e in derived_signs(enc.vitals) + additional_investigation_flags(enc.vitals)
               if not (e.kind == "discriminator" and e.parameter in seen)]
    auto_level = _max_priority(*[e.implies for e in auto_ev])

    result.evidence = disc_ev + auto_ev + tews_ev
    result.missing = missing
    result.ignored_codes = unknown
    if unknown:
        result.warnings.append("Unknown sign code(s) ignored: " + ", ".join(unknown))
    if enc.age_years is None:
        result.warnings.append(
            "Age not recorded — adult pathway assumed (older than 12 years / taller than 150 cm).")
    elif enc.age_years == R.ADULT_MIN_AGE_YEARS:
        result.warnings.append(
            "Patient is 12 — the adult chart applies only if taller than 150 cm; "
            "otherwise use the paediatric chart.")

    sign_codes = {e.parameter for e in disc_ev + auto_ev if e.kind == "discriminator"}
    result.actions = additional_actions(enc, tews_ev, sign_codes)

    tews_band = R.band_for_tews(tews_total) if tews_total is not None else None
    result.tews_total = tews_total
    result.tews_band = tews_band
    result.discriminator_level = _max_priority(disc_level, auto_level)

    proposed = _max_priority(tews_band, disc_level, auto_level)

    # Completeness: we can propose a defensible band if TEWS is complete, OR a
    # discriminator/flag already forces a level regardless of missing vitals.
    forced = _max_priority(disc_level, auto_level)
    if tews_total is None and forced is None:
        result.complete = False
        result.warnings.append(
            "Incomplete: required vitals missing and no overriding sign — "
            "cannot compute a defensible TEWS. Still needed: "
            + ", ".join(param_label_lower(m) for m in missing) + "."
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
        need = ", ".join(param_label_lower(m) for m in missing)
        if proposed == R.Priority.RED:
            result.warnings.append(
                "Emergency sign present — a TEWS is not required before moving to resus. "
                f"Vitals can be completed there (not yet recorded: {need}).")
        else:
            result.warnings.append(
                f"Priority set by a clinical sign; the TEWS is incomplete (not yet recorded: "
                f"{need}). Complete the vitals and re-assess — the priority may rise.")
    return result
