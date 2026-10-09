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


def test_spo2_flag_raises_priority():
    enc = Encounter(encounter_id="E4", vitals=_vit(
        respiratory_rate=12, heart_rate=70, systolic_bp=120, temperature_c=37,
        avpu=AVPU.ALERT, mobility=Mobility.WALKING, trauma=False, spo2=88))
    res = score_encounter(enc)
    assert res.proposed_priority == Priority.ORANGE  # SpO2<92 flag


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
