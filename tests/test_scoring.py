"""Tests that pin the SATS Adult TEWS table and the final-priority logic to the manual.

These are the defensibility tests: they prove each parameter's point allocation matches
the printed Adult SATS Chart, and that scoring is deterministic (same input -> same output).
"""

import pytest

from triageflow.core import sats_rules as R
from triageflow.core.models import Encounter, VitalSigns
from triageflow.core.sats_rules import AVPU, Mobility, Priority
from triageflow.core.scoring import compute_tews, score_encounter, score_vital


def _vit(**kw):
    return VitalSigns(**kw)


# --- per-parameter point allocations (from the Adult TEWS table) ---------------

@pytest.mark.parametrize("rr,pts", [
    (8, 2), (9, 0), (14, 0), (15, 1), (20, 1), (21, 2), (29, 2), (30, 3), (45, 3),
])
def test_rr_points(rr, pts):
    assert score_vital("respiratory_rate", _vit(respiratory_rate=rr)).points == pts


@pytest.mark.parametrize("hr,pts", [
    (40, 2), (41, 1), (50, 1), (51, 0), (100, 0), (101, 1), (110, 1),
    (111, 2), (129, 2), (130, 3),
])
def test_hr_points(hr, pts):
    assert score_vital("heart_rate", _vit(heart_rate=hr)).points == pts


@pytest.mark.parametrize("sbp,pts", [
    (70, 3), (71, 2), (80, 2), (81, 1), (100, 1), (101, 0), (199, 0), (200, 2),
])
def test_sbp_points(sbp, pts):
    assert score_vital("systolic_bp", _vit(systolic_bp=sbp)).points == pts


@pytest.mark.parametrize("t,pts", [
    (34.9, 2), (35, 0), (38.4, 0), (38.5, 2), (40, 2),
])
def test_temp_points(t, pts):
    assert score_vital("temperature_c", _vit(temperature_c=t)).points == pts


@pytest.mark.parametrize("av,pts", [
    (AVPU.ALERT, 0), (AVPU.VOICE, 1), (AVPU.CONFUSED, 2), (AVPU.PAIN, 2),
    (AVPU.UNRESPONSIVE, 3),
])
def test_avpu_points(av, pts):
    assert score_vital("avpu", _vit(avpu=av)).points == pts


@pytest.mark.parametrize("mob,pts", [
    (Mobility.WALKING, 0), (Mobility.WITH_HELP, 1), (Mobility.IMMOBILE, 2),
])
def test_mobility_points(mob, pts):
    assert score_vital("mobility", _vit(mobility=mob)).points == pts


def test_trauma_points():
    assert score_vital("trauma", _vit(trauma=False)).points == 0
    assert score_vital("trauma", _vit(trauma=True)).points == 1


# --- TEWS totals and band boundaries ------------------------------------------

def _healthy_vitals():
    # All-zero TEWS: walking, RR 12, HR 70, SBP 120, Temp 37, Alert, no trauma
    return _vit(respiratory_rate=12, heart_rate=70, systolic_bp=120, temperature_c=37,
                avpu=AVPU.ALERT, mobility=Mobility.WALKING, trauma=False)


def test_healthy_is_green_zero():
    total, ev, missing = compute_tews(_healthy_vitals())
    assert missing == []
    assert total == 0
    assert R.band_for_tews(total) == Priority.GREEN


@pytest.mark.parametrize("total,band", [
    (0, Priority.GREEN), (2, Priority.GREEN), (3, Priority.YELLOW), (4, Priority.YELLOW),
    (5, Priority.ORANGE), (6, Priority.ORANGE), (7, Priority.RED), (12, Priority.RED),
])
def test_band_boundaries(total, band):
    assert R.band_for_tews(total) == band


def test_known_composite_total():
    # HR 125 (+2), RR 26 (+2), SBP 90 (+1), Temp 39 (+2), Voice (+1), with_help (+1), trauma (+1)
    v = _vit(heart_rate=125, respiratory_rate=26, systolic_bp=90, temperature_c=39,
             avpu=AVPU.VOICE, mobility=Mobility.WITH_HELP, trauma=True)
    total, ev, missing = compute_tews(v)
    assert total == 10
    assert R.band_for_tews(total) == Priority.RED


# --- final priority = max(TEWS band, discriminators, investigations) ----------

def test_discriminator_overrides_low_tews():
    enc = Encounter(encounter_id="E1", presenting_complaint="chest pain",
                    vitals=_healthy_vitals(), discriminator_codes=["chest_pain"])
    res = score_encounter(enc)
    assert res.tews_band == Priority.GREEN
    assert res.proposed_priority == Priority.ORANGE  # chest pain = very urgent
    assert res.complete


def test_emergency_sign_forces_red_even_without_full_vitals():
    enc = Encounter(encounter_id="E2", presenting_complaint="collapse",
                    vitals=_vit(heart_rate=80),  # vitals incomplete
                    discriminator_codes=["cardiac_arrest"])
    res = score_encounter(enc)
    assert res.proposed_priority == Priority.RED
    assert res.complete  # forced by sign despite missing vitals


def test_incomplete_without_override_is_not_scored():
    enc = Encounter(encounter_id="E3", vitals=_vit(heart_rate=80))  # missing most params
    res = score_encounter(enc)
    assert res.complete is False
    assert res.proposed_priority is None
    assert "respiratory_rate" in res.missing


@pytest.mark.parametrize("pain,expected", [
    (2, Priority.GREEN),    # mild pain → no escalation
    (5, Priority.YELLOW),   # moderate pain → urgent
    (8, Priority.ORANGE),   # severe pain → very urgent
])
def test_pain_score_maps_to_sats_pain_discriminator(pain, expected):
    enc = Encounter(encounter_id="P1", vitals=_vit(
        respiratory_rate=12, heart_rate=70, systolic_bp=120, temperature_c=37,
        avpu=AVPU.ALERT, mobility=Mobility.WALKING, trauma=False, pain_score=pain))
    res = score_encounter(enc)
    assert res.proposed_priority == expected
    if pain >= 4:  # the pain contribution must be sourced
        assert any("pain" in e.observed.lower() and e.source for e in res.evidence)


def test_diastolic_and_weight_are_recorded_not_scored():
    base = _vit(respiratory_rate=12, heart_rate=70, systolic_bp=120, temperature_c=37,
                avpu=AVPU.ALERT, mobility=Mobility.WALKING, trauma=False)
    withextra = base.model_copy(update={"diastolic_bp": 80, "weight_kg": 72})
    enc_a = Encounter(encounter_id="A", vitals=base)
    enc_b = Encounter(encounter_id="B", vitals=withextra)
    # diastolic/weight do not change TEWS (adult SATS scores systolic only)
    assert score_encounter(enc_a).tews_total == score_encounter(enc_b).tews_total


def test_spo2_floor_raises_priority_and_is_labelled_as_local_not_sats():
    enc = Encounter(encounter_id="E4", age_years=40, vitals=_healthy_vitals().model_copy(
        update={"spo2": 88}))
    res = score_encounter(enc)
    assert res.proposed_priority == Priority.ORANGE
    flag = next(e for e in res.evidence if e.parameter == "spo2")
    # The adult SATS chart has no SpO2 threshold — this must never be cited as SATS.
    assert flag.kind == "safety_rule"
    assert "NOT a SATS adult discriminator" in flag.source
    assert any(a.code == "oxygen_handover" for a in res.actions)


def test_spo2_at_floor_is_not_flagged():
    enc = Encounter(encounter_id="E4b", age_years=40, vitals=_healthy_vitals().model_copy(
        update={"spo2": 92}))
    assert score_encounter(enc).proposed_priority == Priority.GREEN


# --- SATS: "not fully alert" is a very urgent sign, whatever the TEWS total ---------------

@pytest.mark.parametrize("avpu", [AVPU.CONFUSED, AVPU.VOICE, AVPU.PAIN, AVPU.UNRESPONSIVE])
def test_not_fully_alert_is_at_least_orange(avpu):
    enc = Encounter(encounter_id="L1", age_years=50,
                    vitals=_healthy_vitals().model_copy(update={"avpu": avpu}))
    res = score_encounter(enc)
    assert res.tews_band == Priority.GREEN or res.tews_total >= 3   # TEWS alone is low
    assert res.proposed_priority == Priority.ORANGE
    sign = next(e for e in res.evidence if e.parameter == "loc_reduced")
    assert sign.kind == "discriminator" and sign.source == R.SOURCE_LOC


def test_alert_patient_gets_no_loc_sign():
    res = score_encounter(Encounter(encounter_id="L2", age_years=50, vitals=_healthy_vitals()))
    assert not any(e.parameter == "loc_reduced" for e in res.evidence)


def test_confirmed_loc_sign_is_not_duplicated_by_the_derived_one():
    enc = Encounter(encounter_id="L3", age_years=50, discriminator_codes=["loc_reduced"],
                    vitals=_healthy_vitals().model_copy(update={"avpu": AVPU.CONFUSED}))
    res = score_encounter(enc)
    assert len([e for e in res.evidence if e.parameter == "loc_reduced"]) == 1


# --- glucose: 3.0 or less is an emergency; a high reading prompts, never auto-escalates ----

@pytest.mark.parametrize("glucose,expected", [
    (2.9, Priority.RED), (3.0, Priority.RED), (3.1, Priority.GREEN), (28.0, Priority.GREEN)])
def test_glucose_thresholds(glucose, expected):
    enc = Encounter(encounter_id="G1", age_years=50,
                    vitals=_healthy_vitals().model_copy(update={"glucose_mmol": glucose}))
    assert score_encounter(enc).proposed_priority == expected


def test_high_glucose_prompts_ketone_check():
    enc = Encounter(encounter_id="G2", age_years=50,
                    vitals=_healthy_vitals().model_copy(update={"glucose_mmol": 19.0}))
    assert [a.code for a in score_encounter(enc).actions] == ["check_ketones"]


# --- decimals between two printed bands are scored conservatively, never "missing" --------

@pytest.mark.parametrize("param,value,pts", [
    ("respiratory_rate", 14.5, 1), ("respiratory_rate", 20.5, 2), ("respiratory_rate", 8.5, 2),
    ("heart_rate", 100.5, 1), ("heart_rate", 50.5, 1), ("heart_rate", 129.5, 3),
    ("systolic_bp", 80.5, 2), ("systolic_bp", 100.5, 1), ("systolic_bp", 70.5, 3),
    ("systolic_bp", 199.5, 2),
])
def test_decimal_between_bands_takes_the_more_severe_band(param, value, pts):
    ev = score_vital(param, _vit(**{param: value}))
    assert ev is not None and ev.points == pts


def test_decimal_vital_does_not_make_the_tews_incomplete():
    enc = Encounter(encounter_id="D1", age_years=40, vitals=_healthy_vitals().model_copy(
        update={"respiratory_rate": 14.5, "heart_rate": 100.5}))
    res = score_encounter(enc)
    assert res.complete and res.missing == [] and res.tews_total == 2


# --- scope: the adult chart is never applied to a child ----------------------------------

def test_child_is_out_of_scope_and_gets_no_band():
    enc = Encounter(encounter_id="C1", age_years=4, vitals=_healthy_vitals().model_copy(
        update={"heart_rate": 125, "respiratory_rate": 28}))
    res = score_encounter(enc)
    assert res.out_of_scope and not res.complete and res.proposed_priority is None
    assert "paediatric" in res.warnings[0].lower()


def test_missing_age_is_scored_but_flagged():
    res = score_encounter(Encounter(encounter_id="C2", vitals=_healthy_vitals()))
    assert res.complete and any("Age not recorded" in w for w in res.warnings)


# --- unknown sign codes are reported, never silently dropped ------------------------------

def test_unknown_sign_code_is_reported():
    enc = Encounter(encounter_id="U1", age_years=40, vitals=_healthy_vitals(),
                    discriminator_codes=["chest_pian"])
    res = score_encounter(enc)
    assert res.ignored_codes == ["chest_pian"]
    assert any("chest_pian" in w for w in res.warnings)


# --- SATS "additional investigations" are surfaced as actions -----------------------------

def test_actions_follow_the_adult_chart():
    v = _healthy_vitals().model_copy(update={"respiratory_rate": 22, "avpu": AVPU.CONFUSED})
    enc = Encounter(encounter_id="A1", age_years=60, sex="female", vitals=v,
                    presenting_complaint="known diabetic, tummy pain",
                    discriminator_codes=["chest_pain", "abdo_pain"])
    codes = {a.code for a in score_encounter(enc).actions}
    assert codes == {"check_spo2", "check_glucose", "ecg", "pregnancy_test"}
    assert all(a.source for a in score_encounter(enc).actions)


def test_no_pregnancy_test_prompt_for_a_male_patient():
    enc = Encounter(encounter_id="A2", age_years=60, sex="male", vitals=_healthy_vitals(),
                    discriminator_codes=["abdo_pain"])
    assert "pregnancy_test" not in {a.code for a in score_encounter(enc).actions}


def test_emergency_sign_does_not_ask_for_a_tews_first():
    enc = Encounter(encounter_id="A3", age_years=60, vitals=_vit(),
                    discriminator_codes=["cardiac_arrest"])
    res = score_encounter(enc)
    assert res.proposed_priority == Priority.RED
    assert any("TEWS is not required" in w for w in res.warnings)


def test_every_evidence_is_sourced():
    enc = Encounter(encounter_id="E5", vitals=_healthy_vitals(),
                    discriminator_codes=["chest_pain"])
    res = score_encounter(enc)
    assert res.evidence
    assert all(e.source for e in res.evidence)
    assert res.citation_count() == len(res.evidence)


def test_determinism_same_input_same_output():
    enc = Encounter(encounter_id="E6", vitals=_vit(
        heart_rate=125, respiratory_rate=26, systolic_bp=90, temperature_c=39,
        avpu=AVPU.VOICE, mobility=Mobility.WITH_HELP, trauma=True))
    runs = {score_encounter(enc).model_dump_json(exclude={"computed_at"}) for _ in range(20)}
    assert len(runs) == 1  # identical every time
