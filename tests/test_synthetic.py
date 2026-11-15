"""The synthetic patients must be clinically coherent and score to their intended colour."""

import pytest

from triageflow.core.lexicon import suggest_codes
from triageflow.core.sats_rules import DISCRIMINATORS_BY_CODE
from triageflow.core.scoring import score_encounter
from triageflow.core.validation import validate_vitals
from triageflow.data import synthetic
from triageflow.data.synthetic import SCENARIOS, make_scenario

SEEDS = range(60)
# Signs that come from a structured reading (glucose), not from the wording of the note.
FROM_READINGS = {"hypoglycaemia_lt3", "diabetic_gt17", "diabetic_gt11_ketonuria"}


@pytest.mark.parametrize("s", SCENARIOS, ids=lambda s: s.key)
def test_scenario_scores_to_its_expected_colour_for_every_seed(s):
    for seed in SEEDS:
        res = score_encounter(make_scenario(s.key, seed))
        assert res.complete and res.proposed_priority.value == s.expected, (s.key, seed)


@pytest.mark.parametrize("s", SCENARIOS, ids=lambda s: s.key)
def test_scenario_vitals_pass_validation_without_errors_or_extreme_warnings(s):
    for seed in SEEDS:
        e = make_scenario(s.key, seed)
        rep = validate_vitals(e.vitals, now=e.vitals.measured_at, age_years=e.age_years)
        assert rep.issues == [], (s.key, seed, [i.message for i in rep.issues])


@pytest.mark.parametrize("s", SCENARIOS, ids=lambda s: s.key)
def test_the_note_surfaces_the_signs_a_nurse_would_confirm(s):
    """Keyword matching alone (model off) must offer every ground-truth sign of every
    vignette, so the nurse is never left to find it unaided."""
    assert all(c in DISCRIMINATORS_BY_CODE for c in s.codes)
    surfaced = set(suggest_codes(make_scenario(s.key, 1).presenting_complaint))
    assert set(s.codes) - FROM_READINGS <= surfaced, (s.key, surfaced)


@pytest.mark.parametrize("s", [s for s in SCENARIOS if s.expected == "GREEN"], ids=lambda s: s.key)
def test_routine_notes_with_pertinent_negatives_suggest_nothing(s):
    assert suggest_codes(make_scenario(s.key, 1).presenting_complaint) == []


def test_numbers_quoted_in_the_note_are_the_structured_values():
    e = make_scenario("hypoglycaemia", 3)
    assert f"RBS {e.vitals.glucose_mmol:.1f}" in e.presenting_complaint
    e = make_scenario("chest_pain", 3)
    assert f"Pain {e.vitals.pain_score}/10" in e.presenting_complaint


def test_cardiac_arrest_has_no_desk_vitals():
    v = make_scenario("cardiac_arrest", 1).vitals
    assert v.heart_rate is None and v.respiratory_rate is None and v.systolic_bp is None
    assert v.avpu.value == "unresponsive" and v.mobility.value == "immobile"


def test_demographics_fit_the_story():
    for key in ("pv_spotting", "pregnant_abdo", "abdo_pain"):
        for seed in SEEDS:
            e = make_scenario(key, seed)
            assert e.sex == "female" and 16 <= e.age_years <= 45
    for seed in SEEDS:
        for s in SCENARIOS:
            e = make_scenario(s.key, seed)
            assert e.age_years >= 16
            if e.vitals.diastolic_bp is not None:
                assert e.vitals.diastolic_bp < e.vitals.systolic_bp


def test_trauma_flag_matches_the_story():
    assert make_scenario("bp_refill", 1).vitals.trauma is False
    assert make_scenario("rta", 1).vitals.trauma is True


def test_make_encounter_covers_every_acuity_and_is_reproducible():
    for profile, colour in (("stable", "GREEN"), ("moderate", "YELLOW"),
                            ("severe", "ORANGE"), ("critical", "RED")):
        a, b = synthetic.make_encounter(profile, 5), synthetic.make_encounter(profile, 5)
        assert a.presenting_complaint == b.presenting_complaint
        assert a.vitals.heart_rate == b.vitals.heart_rate
        assert score_encounter(a).proposed_priority.value == colour


def test_stream_mixes_acuity_without_repeating_a_story():
    q = synthetic.stream(n=6, seed=42)
    assert len({e.presenting_complaint for e in q}) == 6
    assert {score_encounter(e).proposed_priority.value for e in q} == {"GREEN", "YELLOW", "ORANGE", "RED"}


def test_deterioration_series_gets_worse():
    from triageflow.mcp.server import check_deterioration
    base, recheck = synthetic.deterioration_series(7)
    out = check_deterioration(base.encounter_id, base.vitals.model_dump(mode="json"),
                              recheck.vitals.model_dump(mode="json"))
    assert out["escalated"] is True
