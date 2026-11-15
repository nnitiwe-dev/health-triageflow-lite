"""Tests for validation, FHIR round-trip, deterioration, and determinism of the stream."""

from datetime import UTC, datetime, timedelta

from triageflow.core.models import VitalSigns
from triageflow.core.sats_rules import AVPU, Mobility
from triageflow.core.validation import Severity, validate_vitals
from triageflow.data import fhir, synthetic
from triageflow.mcp.server import check_deterioration


def _ok_vitals(**kw):
    base = dict(respiratory_rate=14, heart_rate=72, systolic_bp=120, temperature_c=36.8,
                avpu=AVPU.ALERT, mobility=Mobility.WALKING, trauma=False)
    base.update(kw)
    return VitalSigns(**base)


def test_validation_passes_clean_vitals():
    assert validate_vitals(_ok_vitals()).ok


def test_validation_flags_missing_required():
    rep = validate_vitals(VitalSigns(heart_rate=80))
    assert "respiratory_rate" in rep.missing_required


def test_validation_impossible_hr_is_error():
    rep = validate_vitals(_ok_vitals(heart_rate=400))
    assert not rep.ok
    assert any(i.field == "heart_rate" and i.severity == Severity.ERROR for i in rep.issues)


def test_extreme_but_possible_values_warn_and_are_not_blocked():
    """RR 3, SBP 35, temp 27.5 are the sickest patients, not typos: warn, never block."""
    for kw in ({"respiratory_rate": 3}, {"systolic_bp": 35}, {"temperature_c": 27.5},
               {"heart_rate": 25}, {"spo2": 60}):
        rep = validate_vitals(_ok_vitals(**kw))
        field = next(iter(kw))
        assert rep.ok, kw
        assert any(i.field == field and i.severity == Severity.WARNING for i in rep.issues), kw


def test_zero_breathing_or_pulse_points_to_the_emergency_sign():
    rr = validate_vitals(_ok_vitals(respiratory_rate=0))
    hr = validate_vitals(_ok_vitals(heart_rate=0))
    assert rr.ok and "not breathing" in rr.issues[0].message.lower()
    assert hr.ok and "cardiac arrest" in hr.issues[0].message.lower()


def test_impossible_values_are_errors():
    for kw in ({"spo2": 140}, {"respiratory_rate": 200}, {"systolic_bp": -5},
               {"temperature_c": 12}):
        assert not validate_vitals(_ok_vitals(**kw)).ok, kw


def test_child_age_is_an_error_naming_the_paediatric_chart():
    rep = validate_vitals(_ok_vitals(), age_years=4)
    assert not rep.ok and "paediatric" in rep.issues[0].message.lower()
    assert validate_vitals(_ok_vitals(), age_years=13).ok


def test_validation_messages_are_plain_language():
    msg = validate_vitals(_ok_vitals(heart_rate=400)).issues[0].message
    assert "Pulse rate" in msg and "heart_rate" not in msg


def test_validation_fahrenheit_temperature_is_error():
    rep = validate_vitals(_ok_vitals(temperature_c=101.0))
    assert not rep.ok
    assert any("Fahrenheit" in i.message for i in rep.issues)


def test_validation_stale_measurement_warns():
    old = datetime.now(UTC) - timedelta(minutes=45)
    rep = validate_vitals(_ok_vitals(measured_at=old))
    assert any(i.field == "measured_at" and i.severity == Severity.WARNING for i in rep.issues)


def test_fhir_bundle_has_expected_resources_and_loinc():
    enc = synthetic.make_encounter("severe", seed=1)
    bundle = fhir.to_bundle(enc)
    kinds = [e["resource"]["resourceType"] for e in bundle["entry"]]
    assert "Patient" in kinds and "Encounter" in kinds and "Observation" in kinds
    hr = next(e["resource"] for e in bundle["entry"]
              if e["resource"]["resourceType"] == "Observation"
              and e["resource"]["code"]["coding"][0].get("code") == "8867-4")
    assert hr["code"]["coding"][0]["system"] == "http://loinc.org"


def test_fhir_round_trip_preserves_vitals():
    enc = synthetic.make_encounter("moderate", seed=3)
    obs = fhir.observations(enc)
    v2 = fhir.vitals_from_observations(obs)
    assert v2.heart_rate == enc.vitals.heart_rate
    assert v2.respiratory_rate == enc.vitals.respiratory_rate
    assert v2.avpu == enc.vitals.avpu


def test_deterioration_tool_detects_escalation():
    base = _ok_vitals(respiratory_rate=22, heart_rate=104).model_dump(mode="json")
    worse = _ok_vitals(respiratory_rate=32, heart_rate=130, systolic_bp=88,
                       avpu=AVPU.VOICE, spo2=85).model_dump(mode="json")
    out = check_deterioration("T1", base, worse)
    assert out["escalated"] is True
    assert out["delta"] >= 2
    assert out["reasons"] and all(r.get("source") for r in out["reasons"])


def test_deterioration_tool_stable_patient_not_flagged():
    base = _ok_vitals().model_dump(mode="json")
    same = _ok_vitals().model_dump(mode="json")
    assert check_deterioration("T2", base, same)["escalated"] is False


def test_deterioration_is_not_silent_when_the_recheck_is_missing_a_vital():
    """A clearly worse re-check with no temperature must still raise the flag."""
    base = _ok_vitals().model_dump(mode="json")
    worse = VitalSigns(respiratory_rate=36, heart_rate=140, systolic_bp=75, avpu=AVPU.PAIN,
                       mobility=Mobility.IMMOBILE, trauma=False).model_dump(mode="json")
    out = check_deterioration("T3", base, worse)
    assert out["escalated"] is True
    assert out["incomplete"] is True and out["missing"] == ["temperature_c"]
    assert "temperature" in out["note"]


def test_deterioration_catches_glucose_pain_and_alertness():
    base = _ok_vitals().model_dump(mode="json")
    for kw, needle in (({"glucose_mmol": 1.8}, "Hypoglycaemia"), ({"pain_score": 9}, "Severe pain"),
                       ({"avpu": AVPU.CONFUSED}, "consciousness")):
        out = check_deterioration("T4", base, _ok_vitals(**kw).model_dump(mode="json"))
        assert out["escalated"] is True, kw
        assert any(needle in r["reason"] for r in out["reasons"]), (kw, out["reasons"])


def test_deterioration_rule_is_not_cited_as_sats():
    base = _ok_vitals().model_dump(mode="json")
    worse = _ok_vitals(respiratory_rate=32, heart_rate=130).model_dump(mode="json")
    out = check_deterioration("T5", base, worse)
    assert any("NOT a SATS rule" in r["source"] for r in out["reasons"])


def test_stream_is_reproducible():
    a = [e.encounter_id for e in synthetic.stream(n=6, seed=42)]
    b = [e.encounter_id for e in synthetic.stream(n=6, seed=42)]
    assert a == b
