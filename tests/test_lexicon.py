"""Lexicon integrity + red-flag sensitivity (recall) tests."""

import pytest

from triageflow.core.lexicon import SYMPTOM_LEXICON, suggest_codes
from triageflow.core.sats_rules import DISCRIMINATORS_BY_CODE


def test_every_lexicon_code_is_a_valid_discriminator():
    for code in SYMPTOM_LEXICON:
        assert code in DISCRIMINATORS_BY_CODE, f"{code} is not a SATS discriminator"


@pytest.mark.parametrize("phrase,expected", [
    ("crushing retrosternal discomfort radiating to jaw", "chest_pain"),  # the old E8 miss
    ("patient is really short of breath and gasping", "sob_acute"),
    ("found collapsed, bystander doing CPR", "cardiac_arrest"),
    ("face drooping and slurred speech since an hour ago", "focal_neuro_acute"),
    ("took an overdose of paracetamol", "poisoning_overdose"),
    ("throwing up blood this morning", "vomiting_fresh_blood"),
    ("fell from height off a roof", "high_energy_transfer"),
    ("broken wrist after tripping", "fracture_closed"),
])
def test_red_flag_phrases_surface_candidate(phrase, expected):
    assert expected in suggest_codes(phrase)


def test_benign_text_surfaces_nothing_urgent():
    assert suggest_codes("here for a routine blood pressure check") == []


@pytest.mark.parametrize("text,code", [
    ("denies chest pain, here for refill", "chest_pain"),
    ("pregnant, no PV bleed noted", "pregnancy_pv_bleed"),
    ("no SOB, no fever", "sob_acute"),
    ("ruled out overdose", "poisoning_overdose"),
])
def test_negated_findings_are_not_surfaced(text, code):
    assert code not in suggest_codes(text)


def test_negation_is_clause_bounded():
    # the negation ("no fever") must not suppress a positive finding in the next clause
    assert "chest_pain" in suggest_codes("no fever, but c/o chest pain radiating to arm")
