"""Symptom lexicon: free-text phrases -> candidate SATS discriminator codes.

This is a deterministic, auditable keyword map used by ``suggest_discriminators``. It only
produces **candidates for the nurse to confirm** — it never auto-applies a sign to a score.

Design notes:
  * Phrases are matched case-insensitively as substrings against the complaint + notes.
  * We bias toward **recall on red-flag presentations** (better to surface a candidate the
    nurse dismisses than to miss one). Precision is handled by the confirm step.
  * Every code here must exist in ``sats_rules.DISCRIMINATORS_BY_CODE``.
  * Keep lay terms AND clinical terms; patients describe symptoms in their own words.

To extend: add phrases to an existing code, or a new ``code: [phrases]`` entry whose code is
a valid SATS discriminator. Add a matching case to ``tests`` / the sensitivity eval.
"""

from __future__ import annotations

import re
from functools import lru_cache

SYMPTOM_LEXICON: dict[str, list[str]] = {
    # --- Emergency (RED) ---
    "cardiac_arrest": [
        "cardiac arrest", "not breathing", "no pulse", "no breathing", "collapsed and not breathing",
        "unresponsive and not breathing", "cpr", "found collapsed",
    ],
    "seizure_active": [
        "seizing", "active seizure", "convulsing", "convulsion", "fitting now", "fit now",
        "jerking", "having a fit", "seizure now",
    ],
    "burn_face_inhalation": [
        "facial burn", "burns to face", "burn to the face", "inhalation burn", "smoke inhalation",
        "burnt airway", "singed", "burns around mouth",
    ],
    "hypoglycaemia_lt3": [
        "very low sugar", "hypo", "hypoglycaemia", "hypoglycemia", "low blood sugar",
    ],

    # --- Very urgent (ORANGE) ---
    "chest_pain": [
        "chest pain", "chest tightness", "chest pressure", "chest discomfort",
        "retrosternal", "behind the sternum", "behind the breastbone", "crushing pain",
        "heavy chest", "tight chest", "pain in my chest", "angina", "radiating to arm",
        "radiating to jaw", "pain spreading to arm",
    ],
    "sob_acute": [
        "short of breath", "shortness of breath", "sob", "difficulty breathing",
        "can't breathe", "cannot breathe", "cant breathe", "struggling to breathe",
        "breathless", "gasping", "wheezing badly", "respiratory distress", "laboured breathing",
    ],
    "loc_reduced": [
        "drowsy", "not making sense", "confused", "altered", "reduced consciousness",
        "hard to rouse", "more sleepy than usual", "disoriented",
    ],
    "coughing_blood": [
        "coughing blood", "coughing up blood", "haemoptysis", "hemoptysis", "blood in sputum",
    ],
    "stabbed_neck": ["stabbed neck", "stab wound to neck", "neck stab", "cut to the neck"],
    "haemorrhage_uncontrolled": [
        "arterial bleed", "spurting blood", "spurting", "uncontrolled bleeding",
        "bleeding heavily", "won't stop bleeding", "wont stop bleeding", "pumping blood",
    ],
    "post_ictal": ["post ictal", "postictal", "after seizure", "after a fit", "drowsy after seizure"],
    "focal_neuro_acute": [
        "facial droop", "face drooping", "stroke", "weakness one side", "one-sided weakness",
        "slurred speech", "can't speak", "arm weakness", "fast positive", "numb one side",
    ],
    "aggression": ["aggressive", "violent", "agitated and aggressive", "combative", "threatening staff"],
    "threatened_limb": [
        "threatened limb", "cold pale limb", "pulseless limb", "blue leg", "blue arm",
        "no pulse in foot",
    ],
    "eye_injury": [
        "eye injury", "something in eye", "eye trauma", "chemical in eye", "penetrating eye",
        "sudden loss of vision", "poked in the eye",
    ],
    "dislocation_large_joint": [
        "dislocated shoulder", "dislocated hip", "dislocated knee", "shoulder out of socket",
        "dislocated elbow",
    ],
    "fracture_compound": [
        "bone sticking out", "open fracture", "compound fracture", "bone through skin",
    ],
    "burn_severe": [
        "large burn", "burn over 20", "electrical burn", "chemical burn", "circumferential burn",
        "extensive burns",
    ],
    "poisoning_overdose": [
        "overdose", "poisoning", "took pills", "took an overdose", "ingested", "swallowed bleach",
        "self poisoning", "od",
    ],
    "vomiting_fresh_blood": [
        "vomiting blood", "vomiting fresh blood", "haematemesis", "hematemesis", "throwing up blood",
    ],
    "pregnancy_abdo": [
        "pregnant and abdominal", "pregnant with abdominal pain", "pregnant abdominal trauma",
        "pregnant and severe pain", "pregnant fall",
    ],
    "severe_pain": [
        "severe pain", "worst pain", "unbearable pain", "10 out of 10 pain", "excruciating",
        "agonising pain", "agonizing pain",
    ],
    "high_energy_transfer": [
        "high speed", "car rollover", "ejected from vehicle", "fall from height",
        "fell from height", "from a height", "off a roof", "off the roof", "motorbike crash",
        "pedestrian hit", "high energy", "severe mechanism",
    ],

    # --- Urgent (YELLOW) ---
    "haemorrhage_controlled": ["controlled bleeding", "bleeding now stopped", "dressing holding"],
    "dislocation_finger_toe": ["dislocated finger", "dislocated toe", "finger out of joint"],
    "fracture_closed": [
        "closed fracture", "broken arm", "broken leg", "broken wrist", "broken ankle",
        "possible fracture", "fracture", "suspected break",
    ],
    "burn_other": ["minor burn", "small burn", "scald", "burnt hand"],
    "abdo_pain": [
        "abdominal pain", "stomach pain", "belly pain", "tummy pain", "stomach ache",
        "pain in my stomach", "abdo pain",
    ],
    "vomiting_persistent": [
        "persistent vomiting", "vomiting all day", "keeps vomiting", "can't stop vomiting",
        "throwing up repeatedly",
    ],
    "pregnancy_trauma": ["pregnant and fell", "pregnant after accident", "pregnant trauma"],
    "pregnancy_pv_bleed": [
        "pv bleed", "vaginal bleeding pregnant", "bleeding and pregnant", "spotting pregnant",
    ],
    "moderate_pain": ["moderate pain", "bad pain", "quite painful", "painful but bearable"],
}


@lru_cache(maxsize=512)
def _pattern(phrase: str) -> re.Pattern[str]:
    # Match the phrase only when not surrounded by other letters/digits, so short tokens
    # like "od" don't match inside "blood". Internal spaces/punctuation are matched literally.
    return re.compile(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", re.IGNORECASE)


def suggest_codes(text: str) -> list[str]:
    """Return candidate discriminator codes whose phrases appear in ``text`` (deduped, ordered).

    Uses word-boundary matching to avoid false positives from short abbreviations.
    """
    t = (text or "").lower()
    hits: list[str] = []
    for code, phrases in SYMPTOM_LEXICON.items():
        if any(_pattern(p).search(t) for p in phrases):
            hits.append(code)
    return hits
