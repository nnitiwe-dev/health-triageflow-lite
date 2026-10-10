"""Optional FHIR / EMR integration (OFF unless ``TRIAGEFLOW_FHIR_BASE_URL`` is set).

Lets a nurse **import a patient's vitals from a FHIR R4 server** instead of re-typing them —
a real "fit to the desk" win — and proves interoperability with a genuine write→read round
trip. All data is synthetic; point the base URL at a facility's IN-COUNTRY FHIR server for a
pilot so patient data never leaves the country. Everything degrades gracefully: if the server
is unreachable, the manual/synthetic path is unaffected.
"""

from __future__ import annotations

import uuid
from typing import Any

import httpx

from ..core.config import settings
from ..core.models import Encounter, VitalSigns
from ..data import fhir as fhirbuild

_FHIR_HEADERS = {"Accept": "application/fhir+json", "Content-Type": "application/fhir+json"}


def fhir_enabled() -> bool:
    return bool(settings.fhir_base_url)


def _client(timeout: float = 20.0) -> httpx.Client:
    return httpx.Client(base_url=settings.fhir_base_url, timeout=timeout, headers=_FHIR_HEADERS)


class FHIRUnavailable(RuntimeError):
    """Raised when the configured FHIR server can't be reached — caller falls back."""


def _transaction_bundle(enc: Encounter) -> dict[str, Any]:
    """Build a FHIR transaction Bundle (Patient + Observations) with urn:uuid references,
    so a server assigns real ids and links them atomically."""
    pat_urn = f"urn:uuid:{uuid.uuid4()}"
    patient = fhirbuild.patient_resource(enc)
    patient.pop("id", None)
    entries: list[dict[str, Any]] = [{
        "fullUrl": pat_urn, "resource": patient,
        "request": {"method": "POST", "url": "Patient"},
    }]
    for obs in fhirbuild.observations(enc):
        obs.pop("id", None)
        obs["subject"] = {"reference": pat_urn}
        obs.pop("encounter", None)
        entries.append({"fullUrl": f"urn:uuid:{uuid.uuid4()}", "resource": obs,
                        "request": {"method": "POST", "url": "Observation"}})
    return {"resourceType": "Bundle", "type": "transaction", "entry": entries}


def seed_patient(enc: Encounter) -> str:
    """Write a synthetic patient + vitals to the FHIR server; return the created Patient id."""
    bundle = _transaction_bundle(enc)
    try:
        with _client() as c:
            r = c.post("/", json=bundle)
            r.raise_for_status()
            resp = r.json()
    except (httpx.HTTPError, ValueError) as e:
        raise FHIRUnavailable(f"seed failed: {type(e).__name__}: {e}") from e
    # The first entry's response location is the created Patient (e.g. "Patient/123/_history/1")
    for entry in resp.get("entry", []):
        loc = entry.get("response", {}).get("location", "")
        if loc.startswith("Patient/"):
            return loc.split("/")[1]
    raise FHIRUnavailable("seed succeeded but no Patient id returned")


def fetch_observations(patient_id: str) -> list[dict[str, Any]]:
    try:
        with _client() as c:
            r = c.get("/Observation", params={"subject": f"Patient/{patient_id}", "_count": "100"})
            r.raise_for_status()
            bundle = r.json()
    except (httpx.HTTPError, ValueError) as e:
        raise FHIRUnavailable(f"fetch failed: {type(e).__name__}: {e}") from e
    return [e["resource"] for e in bundle.get("entry", []) if e.get("resource")]


def import_patient_vitals(patient_id: str) -> VitalSigns:
    """Read a patient's Observations from the FHIR server and map them to VitalSigns."""
    return fhirbuild.vitals_from_observations(fetch_observations(patient_id))


def demo_round_trip(enc: Encounter) -> tuple[str, VitalSigns]:
    """Seed one synthetic patient, then read their vitals back — a full FHIR round trip.
    Returns (patient_id, vitals_read_back)."""
    pid = seed_patient(enc)
    return pid, import_patient_vitals(pid)


def server_status() -> dict[str, Any]:
    """Lightweight reachability check for the UI (does not write anything)."""
    if not fhir_enabled():
        return {"enabled": False, "reachable": False, "base_url": None}
    try:
        with _client(timeout=8) as c:
            r = c.get("/metadata", params={"_summary": "true"})
            ok = r.status_code == 200
    except httpx.HTTPError:
        ok = False
    return {"enabled": True, "reachable": ok, "base_url": settings.fhir_base_url}
