"""Symptom lexicon: free-text phrases -> candidate SATS discriminator codes.

This is a deterministic, auditable keyword map used by ``suggest_discriminators``. It only
produces **candidates for the nurse to confirm** — it never auto-applies a sign to a score.

Design notes:
  * Phrases are matched case-insensitively on word boundaries against the complaint + notes.
  * A phrase written ``"a + b"`` matches only when BOTH parts appear (e.g. pregnancy + bleeding).
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
        "cardiac arrest", "no pulse", "pulseless", "cpr", "collapsed and not breathing",
        "unresponsive and not breathing", "arrested",
    ],
    "airway_obstructed": [
        "not breathing", "no breathing", "stopped breathing", "obstructed airway", "choking",
        "airway obstruction", "cannot breathe at all", "apnoeic", "apneic",
    ],
    "seizure_active": [
        "seizing", "active seizure", "convulsing", "convulsion", "fitting now", "fit now",
        "jerking", "having a fit", "seizure now", "still fitting", "still convulsing",
        "actively seizing", "status epilepticus", "degedege",
    ],
    "burn_face_inhalation": [
        "facial burn", "burns to face", "burn to the face", "inhalation burn", "smoke inhalation",
        "burnt airway", "singed", "burns around mouth", "soot around mouth", "soot in mouth",
        "hoarse voice after fire", "steam inhalation burn",
    ],
    "hypoglycaemia_lt3": [
        "very low sugar", "hypo", "hypoglycaemia", "hypoglycemia", "low blood sugar",
    ],

    # --- Very urgent (ORANGE) ---
    "chest_pain": [
        "chest pain", "chest tightness", "chest pressure", "chest discomfort",
        "retrosternal", "behind the sternum", "behind the breastbone", "crushing pain",
        "heavy chest", "tight chest", "pain in my chest", "angina", "radiating to arm",
        "radiating to jaw", "pain spreading to arm", "heart attack", "pain in chest",
        "pain in the chest", "chest heaviness", "chest is paining", "chest dey pain",
        "maumivu ya kifua",
    ],
    "sob_acute": [
        "short of breath", "shortness of breath", "sob", "difficulty breathing",
        "can't breathe", "cannot breathe", "cant breathe", "struggling to breathe",
        "breathless", "gasping", "wheezing badly", "respiratory distress", "laboured breathing",
        "difficulty in breathing", "difficulty with breathing", "breathing difficulty",
        "trouble breathing", "can not breathe", "cannot breath", "can't breath",
        "not breathing well", "fast breathing", "dyspnoea", "dyspnea", "asthma attack",
        "wheezing", "speaking in short sentences", "no fit breathe", "kushindwa kupumua",
    ],
    "loc_reduced": [
        "drowsy", "not making sense", "confused", "altered", "reduced consciousness",
        "hard to rouse", "more sleepy than usual", "disoriented", "found collapsed",
        "collapsed", "fainted", "passed out", "syncope", "loss of consciousness", "unconscious",
        "unresponsive", "not responding", "lethargic", "confusion",
    ],
    "coughing_blood": [
        "coughing blood", "coughing up blood", "haemoptysis", "hemoptysis", "blood in sputum",
        "coughed blood", "coughed up blood", "blood-stained sputum", "cough with blood",
    ],
    "stabbed_neck": ["stabbed neck", "stab wound to neck", "neck stab", "cut to the neck"],
    "haemorrhage_uncontrolled": [
        "arterial bleed", "spurting blood", "spurting", "uncontrolled bleeding",
        "bleeding heavily", "won't stop bleeding", "wont stop bleeding", "pumping blood",
        "heavy bleeding", "bleeding profusely", "profuse bleeding", "soaked through dressing",
    ],
    "post_ictal": [
        "post ictal", "postictal", "post-ictal", "after seizure", "after a fit",
        "drowsy after seizure", "convulsed", "had a fit", "had a seizure", "seizure at home",
        "fitted at home", "seizure today", "convulsion at home",
    ],
    "focal_neuro_acute": [
        "facial droop", "face drooping", "stroke", "weakness one side", "one-sided weakness",
        "slurred speech", "can't speak", "arm weakness", "fast positive", "numb one side",
        "weak on left side", "weak on right side", "weakness on one side", "left sided weakness",
        "right sided weakness", "left-sided weakness", "right-sided weakness", "arm weak",
        "mouth twisted", "twisted mouth", "mouth deviation", "facial deviation",
        "unable to talk", "unable to speak", "cannot speak", "hemiparesis", "hemiplegia", "cva",
    ],
    "aggression": ["aggressive", "violent", "agitated and aggressive", "combative",
                   "threatening staff", "aggression", "fighting staff"],
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
        "bone exposed", "bone visible",
    ],
    "burn_severe": [
        "large burn", "burn over 20", "electrical burn", "chemical burn", "circumferential burn",
        "extensive burns",
    ],
    "poisoning_overdose": [
        "overdose", "poisoning", "poison", "poisoned", "took pills", "took tablets",
        "took an overdose", "ingested", "swallowed bleach", "self poisoning", "o/d",
        "organophosphate", "pesticide", "kerosene", "paraffin ingestion", "took sniper",
        "drank bleach", "drank kerosene", "took + tablets", "swallowed + tablets",
        "took + pills",
    ],
    "vomiting_fresh_blood": [
        "vomiting blood", "vomiting fresh blood", "haematemesis", "hematemesis", "throwing up blood",
        "vomited blood", "vomited fresh blood", "vomiting fresh blood", "blood in vomit",
    ],
    "pregnancy_abdo": [
        "pregnant and abdominal", "pregnant with abdominal pain", "pregnant abdominal trauma",
        "pregnant and severe pain", "pregnant fall", "pregnant + abdominal pain",
        "pregnant + abdo pain", "pregnant + stomach pain", "pregnant + lower abdominal",
        "pregnancy + abdominal pain", "gravid + abdominal pain",
    ],
    "severe_pain": [
        "severe pain", "worst pain", "unbearable pain", "10 out of 10 pain", "excruciating",
        "agonising pain", "agonizing pain", "worst headache", "worst of my life",
        "pain 10/10", "10/10 pain", "9/10 pain", "pain 9/10",
    ],
    "high_energy_transfer": [
        "high speed", "car rollover", "ejected from vehicle", "fall from height",
        "fell from height", "from a height", "off a roof", "off the roof", "motorbike crash",
        "pedestrian hit", "high energy", "severe mechanism", "rta", "rtc", "mva", "mvc",
        "road traffic accident", "road accident", "motor vehicle accident", "knocked down",
        "hit by a car", "hit by car", "hit by a vehicle", "boda boda", "okada accident",
        "motorcycle accident", "thrown off", "gunshot", "gsw", "fell from a tree",
        "fell from tree", "fell from a roof", "fell from roof", "fell off a roof",
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
        "pain in my stomach", "abdo pain", "abdominal pains", "stomach pains",
        "lower abdominal pain", "epigastric pain", "belle pain", "belle dey pain",
        "maumivu ya tumbo",
    ],
    "vomiting_persistent": [
        "persistent vomiting", "vomiting all day", "keeps vomiting", "can't stop vomiting",
        "throwing up repeatedly", "vomiting everything", "vomited several times",
        "vomits everything", "vomiting since", "not keeping fluids down",
        "not keeping anything down", "unable to keep fluids down",
    ],
    "pregnancy_trauma": ["pregnant and fell", "pregnant after accident", "pregnant trauma",
                         "pregnant + fell", "pregnant + fall", "pregnant + accident",
                         "pregnant + assault", "pregnant + rta"],
    "pregnancy_pv_bleed": [
        "pv bleed", "vaginal bleeding pregnant", "bleeding and pregnant", "spotting pregnant",
        "pregnant + bleeding", "pregnant + pv bleed", "pregnant + spotting",
        "pregnancy + bleeding", "gravid + bleeding", "antenatal + bleeding",
    ],
    "moderate_pain": ["moderate pain", "bad pain", "quite painful", "painful but bearable"],
}


@lru_cache(maxsize=512)
def _pattern(phrase: str) -> re.Pattern[str]:
    # Match the phrase only when not surrounded by other letters/digits, so short tokens
    # like "od" don't match inside "blood". Internal spaces/punctuation are matched literally.
    return re.compile(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", re.IGNORECASE)


# Negation cues that, appearing just before a phrase (within the same clause), mean the
# finding is a pertinent NEGATIVE and must not be surfaced — a classic triage-NLP pitfall
# (e.g. "no PV bleed", "denies chest pain"). Clause-bounded so a comma/period resets it.
_NEGATION = re.compile(
    r"(?:\b(?:no|not|without|denies|denied|neg|ruled out|free of|nil)\b)[^.,;]{0,24}$",
    re.IGNORECASE,
)


def _is_negated(text: str, start: int) -> bool:
    return bool(_NEGATION.search(text[:start]))


def _matches(phrase: str, t: str) -> bool:
    """True if ``phrase`` is present (and not negated). ``"a + b"`` needs every part present."""
    for part in phrase.split(" + "):
        found = False
        for m in _pattern(part).finditer(t):
            if not _is_negated(t, m.start()):
                found = True
                break
        if not found:
            return False
    return True


def suggest_codes(text: str) -> list[str]:
    """Return candidate discriminator codes whose phrases appear in ``text`` (deduped, ordered).

    Word-boundary matching avoids false positives from short abbreviations; a light negation
    check drops findings the note explicitly denies (so "no PV bleed" is not flagged).
    """
    t = (text or "").lower()
    return [code for code, phrases in SYMPTOM_LEXICON.items()
            if any(_matches(p, t) for p in phrases)]
