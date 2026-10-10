"""Network-free tests for the FHIR/EMR integration (transaction bundle + mapping)."""

from triageflow.agent import fhir_emr
from triageflow.data import fhir as fhirbuild
from triageflow.data import synthetic


def test_disabled_by_default(monkeypatch):
    # With no base URL configured, the integration is off and never touches the network.
    monkeypatch.setattr(fhir_emr.settings, "fhir_base_url", "")
    assert fhir_emr.fhir_enabled() is False
    assert fhir_emr.server_status() == {"enabled": False, "reachable": False, "base_url": None}


def test_transaction_bundle_structure():
    enc = synthetic.make_encounter("severe", seed=5)
    bundle = fhir_emr._transaction_bundle(enc)
    assert bundle["type"] == "transaction"
    methods = [e["request"]["method"] for e in bundle["entry"]]
    assert set(methods) == {"POST"}
    types = [e["resource"]["resourceType"] for e in bundle["entry"]]
    assert types[0] == "Patient"
    assert "Observation" in types
    # Every observation references the Patient by its urn:uuid fullUrl (atomic linkage)
    pat_urn = bundle["entry"][0]["fullUrl"]
    assert pat_urn.startswith("urn:uuid:")
    for e in bundle["entry"][1:]:
        assert e["resource"]["subject"]["reference"] == pat_urn


def test_observations_round_trip_via_mapping():
    # The server stores then returns Observations; mapping must recover the vitals.
    enc = synthetic.make_encounter("moderate", seed=9)
    obs = [e["resource"] for e in fhir_emr._transaction_bundle(enc)["entry"]
           if e["resource"]["resourceType"] == "Observation"]
    v = fhirbuild.vitals_from_observations(obs)
    assert v.heart_rate == enc.vitals.heart_rate
    assert v.respiratory_rate == enc.vitals.respiratory_rate
    assert v.avpu == enc.vitals.avpu
