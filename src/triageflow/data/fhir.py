"""Minimal FHIR R4 builders — makes synthetic encounters EMR/EHR-shaped.

Vitals use **LOINC** codes (free to use) so the data looks like what a real FHIR/EMR
API would return. AVPU, mobility, trauma and presenting complaint use a small
**self-authored** code system (``CS_LOCAL``) so we ship zero restricted ontology
(no SNOMED CT redistribution — that needs an Affiliate licence and carries fees in
non-member countries).

These are plain dicts (valid FHIR R4 JSON), not a FHIR SDK, to keep deps light.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ..core.models import Encounter, VitalSigns

CS_LOCAL = "https://triageflow.local/fhir/CodeSystem/triage-obs"

# LOINC codes for the standard vital signs.
LOINC = {
    "respiratory_rate": ("9279-1", "Respiratory rate", "/min"),
    "heart_rate": ("8867-4", "Heart rate", "/min"),
    "systolic_bp": ("8480-6", "Systolic blood pressure", "mm[Hg]"),
    "temperature_c": ("8310-5", "Body temperature", "Cel"),
    "spo2": ("2708-6", "Oxygen saturation", "%"),
    "glucose_mmol": ("15074-8", "Glucose [Moles/volume] in Blood", "mmol/L"),
}
# Non-standard observations -> local code system.
LOCAL_CODES = {
    "avpu": ("avpu", "AVPU responsiveness"),
    "mobility": ("mobility", "Mobility at triage"),
    "trauma": ("trauma", "Trauma-related presentation"),
}


def _ref(resource: str, rid: str) -> dict[str, str]:
    return {"reference": f"{resource}/{rid}"}


def _coding(system: str, code: str, display: str) -> dict[str, Any]:
    return {"coding": [{"system": system, "code": code, "display": display}]}


def patient_resource(enc: Encounter) -> dict[str, Any]:
    return {
        "resourceType": "Patient",
        "id": f"pat-{enc.encounter_id}",
        "gender": (enc.sex or "unknown"),
        "extension": [{
            "url": "http://hl7.org/fhir/StructureDefinition/patient-age",
            "valueInteger": enc.age_years,
        }] if enc.age_years is not None else [],
    }


def _quantity_obs(enc: Encounter, param: str, value: float) -> dict[str, Any]:
    code, display, unit = LOINC[param]
    return {
        "resourceType": "Observation",
        "id": f"obs-{enc.encounter_id}-{param}",
        "status": "final",
        "category": [_coding("http://terminology.hl7.org/CodeSystem/observation-category",
                             "vital-signs", "Vital Signs")],
        "code": _coding("http://loinc.org", code, display),
        "subject": _ref("Patient", f"pat-{enc.encounter_id}"),
        "encounter": _ref("Encounter", f"enc-{enc.encounter_id}"),
        "effectiveDateTime": enc.vitals.measured_at.isoformat(),
        "valueQuantity": {"value": value, "unit": unit, "system": "http://unitsofmeasure.org",
                          "code": unit},
    }


def _coded_obs(enc: Encounter, param: str, code: str, display: str) -> dict[str, Any]:
    local_code, local_display = LOCAL_CODES[param]
    return {
        "resourceType": "Observation",
        "id": f"obs-{enc.encounter_id}-{param}",
        "status": "final",
        "code": _coding(CS_LOCAL, local_code, local_display),
        "subject": _ref("Patient", f"pat-{enc.encounter_id}"),
        "encounter": _ref("Encounter", f"enc-{enc.encounter_id}"),
        "effectiveDateTime": enc.vitals.measured_at.isoformat(),
        "valueCodeableConcept": _coding(CS_LOCAL, code, display),
    }


def observations(enc: Encounter) -> list[dict[str, Any]]:
    v = enc.vitals
    out: list[dict[str, Any]] = []
    for param in ("respiratory_rate", "heart_rate", "systolic_bp", "temperature_c",
                  "spo2", "glucose_mmol"):
        val = getattr(v, param)
        if val is not None:
            out.append(_quantity_obs(enc, param, val))
    if v.avpu is not None:
        out.append(_coded_obs(enc, "avpu", v.avpu.value, v.avpu.value))
    if v.mobility is not None:
        out.append(_coded_obs(enc, "mobility", v.mobility.value, v.mobility.value))
    if v.trauma is not None:
        out.append(_coded_obs(enc, "trauma", str(v.trauma).lower(),
                              "trauma" if v.trauma else "no trauma"))
    return out


def condition_resource(enc: Encounter) -> dict[str, Any] | None:
    if not enc.presenting_complaint and not enc.discriminator_codes:
        return None
    codes = [{"system": CS_LOCAL, "code": c, "display": c} for c in enc.discriminator_codes]
    return {
        "resourceType": "Condition",
        "id": f"cond-{enc.encounter_id}",
        "clinicalStatus": _coding(
            "http://terminology.hl7.org/CodeSystem/condition-clinical", "active", "Active"),
        "category": [_coding(
            "http://terminology.hl7.org/CodeSystem/condition-category",
            "encounter-diagnosis", "Encounter Diagnosis")],
        "code": {"text": enc.presenting_complaint, "coding": codes},
        "subject": _ref("Patient", f"pat-{enc.encounter_id}"),
        "encounter": _ref("Encounter", f"enc-{enc.encounter_id}"),
    }


def encounter_resource(enc: Encounter) -> dict[str, Any]:
    return {
        "resourceType": "Encounter",
        "id": f"enc-{enc.encounter_id}",
        "status": "in-progress",
        "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                  "code": "EMER", "display": "emergency"},
        "subject": _ref("Patient", f"pat-{enc.encounter_id}"),
        "period": {"start": enc.arrival_time.isoformat()},
        "reasonCode": [{"text": enc.presenting_complaint}] if enc.presenting_complaint else [],
    }


def to_bundle(enc: Encounter) -> dict[str, Any]:
    """A FHIR R4 transaction-style Bundle for one encounter."""
    entries: list[dict[str, Any]] = [
        {"resource": patient_resource(enc)},
        {"resource": encounter_resource(enc)},
    ]
    cond = condition_resource(enc)
    if cond:
        entries.append({"resource": cond})
    entries.extend({"resource": o} for o in observations(enc))
    return {"resourceType": "Bundle", "type": "collection",
            "timestamp": datetime.now(UTC).isoformat(), "entry": entries}


def vitals_from_observations(obs: list[dict[str, Any]]) -> VitalSigns:
    """Inverse: parse a list of FHIR Observations back into VitalSigns (EMR ingest path)."""
    by_loinc = {v[0]: k for k, v in LOINC.items()}
    kw: dict[str, Any] = {}
    for o in obs:
        coding = (o.get("code", {}).get("coding") or [{}])[0]
        system, code = coding.get("system"), coding.get("code")
        if system == "http://loinc.org" and code in by_loinc:
            kw[by_loinc[code]] = o.get("valueQuantity", {}).get("value")
        elif system == CS_LOCAL:
            val = (o.get("valueCodeableConcept", {}).get("coding") or [{}])[0].get("code")
            if code == "avpu":
                kw["avpu"] = val
            elif code == "mobility":
                kw["mobility"] = val
            elif code == "trauma":
                kw["trauma"] = (val == "true")
    return VitalSigns(**kw)
