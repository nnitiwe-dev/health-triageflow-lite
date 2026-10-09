"""SATS Adult rule table — machine-readable, each rule carries its source citation.

This module encodes clinical *facts* (thresholds and discriminators) from the
South African Triage Scale Adult pathway. It is the single source of truth for the
deterministic scoring engine in ``scoring.py``.

Attribution: South African Triage Group (SATG) / EMSSA, *SATS Training Manual 2012*.
SATS materials are CC BY-NC-SA 3.0; this project ships only a re-encoded rule table
(facts) and cites the source — it does not redistribute the SATS charts or manual.

Nothing here diagnoses. It produces a *sourced priority level* for a clinician to confirm.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

# Version of the encoded ruleset — bump when thresholds change, for auditability.
RULESET_VERSION = "sats-adult-2012.1"
SOURCE_CHART = "SATS Training Manual 2012, Adult SATS Chart (Figure 4, p. 8-9)"
SOURCE_TEWS = "SATS Training Manual 2012, Adult SATS Chart 'ADULT TEWS' table (Figure 4, p. 9)"
SOURCE_FLOW = "SATS Training Manual 2012, Adult SATS process flowchart (Figure 2/4)"
SOURCE_LEVELS = "SATS Training Manual 2012, SATS priority levels and target times (Table 3)"


class Priority(str, Enum):
    """SATS priority colours, ordered by acuity (see ``ACUITY``)."""

    BLUE = "BLUE"      # deceased
    GREEN = "GREEN"    # routine
    YELLOW = "YELLOW"  # urgent
    ORANGE = "ORANGE"  # very urgent
    RED = "RED"        # emergency


# Higher number = more acute. Used to take the max across TEWS band and discriminators.
ACUITY: dict[Priority, int] = {
    Priority.BLUE: -1,
    Priority.GREEN: 0,
    Priority.YELLOW: 1,
    Priority.ORANGE: 2,
    Priority.RED: 3,
}

TARGET_TIME: dict[Priority, str] = {
    Priority.RED: "Immediate",
    Priority.ORANGE: "< 10 minutes",
    Priority.YELLOW: "< 1 hour",
    Priority.GREEN: "< 4 hours",
    Priority.BLUE: "< 2 hours (certification)",
}

DISPOSITION: dict[Priority, str] = {
    Priority.RED: "Resuscitation area",
    Priority.ORANGE: "Majors (very urgent)",
    Priority.YELLOW: "Majors (urgent)",
    Priority.GREEN: "Designated non-urgent area",
    Priority.BLUE: "Refer for certification",
}


class AVPU(str, Enum):
    ALERT = "alert"
    VOICE = "voice"            # reacts to voice
    PAIN = "pain"              # reacts to pain
    UNRESPONSIVE = "unresponsive"
    CONFUSED = "confused"


class Mobility(str, Enum):
    WALKING = "walking"
    WITH_HELP = "with_help"
    IMMOBILE = "immobile"      # stretcher / immobile


# --- TEWS parameter bands ------------------------------------------------------
# Each band: (points, human label, predicate). Evaluated in order; first match wins.


@dataclass(frozen=True)
class Band:
    points: int
    label: str
    test: Callable[[float], bool]


# Respiratory rate (breaths/min): <9→2, 9-14→0, 15-20→1, 21-29→2, >29→3
RR_BANDS: list[Band] = [
    Band(2, "< 9", lambda v: v < 9),
    Band(0, "9-14", lambda v: 9 <= v <= 14),
    Band(1, "15-20", lambda v: 15 <= v <= 20),
    Band(2, "21-29", lambda v: 21 <= v <= 29),
    Band(3, "> 29", lambda v: v > 29),
]

# Heart rate (bpm): <41→2, 41-50→1, 51-100→0, 101-110→1, 111-129→2, >129→3
HR_BANDS: list[Band] = [
    Band(2, "< 41", lambda v: v < 41),
    Band(1, "41-50", lambda v: 41 <= v <= 50),
    Band(0, "51-100", lambda v: 51 <= v <= 100),
    Band(1, "101-110", lambda v: 101 <= v <= 110),
    Band(2, "111-129", lambda v: 111 <= v <= 129),
    Band(3, "> 129", lambda v: v > 129),
]

# Systolic BP (mmHg): <71→3, 71-80→2, 81-100→1, 101-199→0, >199→2
SBP_BANDS: list[Band] = [
    Band(3, "< 71", lambda v: v < 71),
    Band(2, "71-80", lambda v: 71 <= v <= 80),
    Band(1, "81-100", lambda v: 81 <= v <= 100),
    Band(0, "101-199", lambda v: 101 <= v <= 199),
    Band(2, "> 199", lambda v: v > 199),
]

# Temperature (°C): <35→2, 35-38.4→0, >38.4→2
TEMP_BANDS: list[Band] = [
    Band(2, "< 35", lambda v: v < 35),
    Band(0, "35-38.4", lambda v: 35 <= v <= 38.4),
    Band(2, "> 38.4", lambda v: v > 38.4),
]

# Categorical parameters: value -> (points, label)
MOBILITY_POINTS: dict[Mobility, tuple[int, str]] = {
    Mobility.WALKING: (0, "Walking"),
    Mobility.WITH_HELP: (1, "With help"),
    Mobility.IMMOBILE: (2, "Stretcher / immobile"),
}

AVPU_POINTS: dict[AVPU, tuple[int, str]] = {
    AVPU.ALERT: (0, "Alert"),
    AVPU.VOICE: (1, "Reacts to voice"),
    AVPU.CONFUSED: (2, "Confused"),
    AVPU.PAIN: (2, "Reacts to pain"),
    AVPU.UNRESPONSIVE: (3, "Unresponsive"),
}

TRAUMA_POINTS: dict[bool, tuple[int, str]] = {
    False: (0, "No"),
    True: (1, "Yes"),
}


# --- TEWS total -> band --------------------------------------------------------

def band_for_tews(total: int) -> Priority:
    """Map a TEWS total to its SATS colour band. Source: {SOURCE_FLOW}."""
    if total >= 7:
        return Priority.RED
    if total >= 5:
        return Priority.ORANGE
    if total >= 3:
        return Priority.YELLOW
    return Priority.GREEN


# --- Discriminators ------------------------------------------------------------


@dataclass(frozen=True)
class Discriminator:
    code: str           # our own stable slug
    label: str
    level: Priority     # minimum priority this discriminator implies
    source: str = SOURCE_CHART


# Emergency signs -> RED
EMERGENCY_SIGNS: list[Discriminator] = [
    Discriminator("airway_obstructed", "Obstructed airway / not breathing", Priority.RED),
    Discriminator("seizure_active", "Current (active) seizure", Priority.RED),
    Discriminator("burn_face_inhalation", "Facial or inhalation burn", Priority.RED),
    Discriminator("hypoglycaemia_lt3", "Hypoglycaemia (glucose < 3 mmol/L)", Priority.RED),
    Discriminator("cardiac_arrest", "Cardiac arrest", Priority.RED),
]

# Very urgent signs -> ORANGE
VERY_URGENT_SIGNS: list[Discriminator] = [
    Discriminator("sob_acute", "Acute shortness of breath", Priority.ORANGE),
    Discriminator("loc_reduced", "Reduced / confused level of consciousness", Priority.ORANGE),
    Discriminator("coughing_blood", "Coughing blood", Priority.ORANGE),
    Discriminator("chest_pain", "Chest pain", Priority.ORANGE),
    Discriminator("stabbed_neck", "Stabbed neck", Priority.ORANGE),
    Discriminator("haemorrhage_uncontrolled", "Uncontrolled haemorrhage (arterial)", Priority.ORANGE),
    Discriminator("post_ictal", "Post-ictal (after seizure)", Priority.ORANGE),
    Discriminator("focal_neuro_acute", "Acute focal neurology (stroke)", Priority.ORANGE),
    Discriminator("aggression", "Aggression", Priority.ORANGE),
    Discriminator("threatened_limb", "Threatened limb", Priority.ORANGE),
    Discriminator("eye_injury", "Eye injury", Priority.ORANGE),
    Discriminator("dislocation_large_joint", "Dislocation of a larger joint", Priority.ORANGE),
    Discriminator("fracture_compound", "Compound fracture (break in skin)", Priority.ORANGE),
    Discriminator("burn_severe", "Burn >20% / electrical / circumferential / chemical", Priority.ORANGE),
    Discriminator("poisoning_overdose", "Poisoning / overdose", Priority.ORANGE),
    Discriminator("diabetic_gt11_ketonuria", "Diabetic glucose >11 with ketonuria", Priority.ORANGE),
    Discriminator("vomiting_fresh_blood", "Vomiting fresh blood", Priority.ORANGE),
    Discriminator("pregnancy_abdo", "Pregnancy with abdominal trauma or pain", Priority.ORANGE),
    Discriminator("severe_pain", "Severe pain", Priority.ORANGE),
    Discriminator("high_energy_transfer", "High-energy transfer (severe mechanism)", Priority.ORANGE),
]

# Urgent signs -> YELLOW
URGENT_SIGNS: list[Discriminator] = [
    Discriminator("haemorrhage_controlled", "Controlled haemorrhage", Priority.YELLOW),
    Discriminator("dislocation_finger_toe", "Dislocation of finger or toe", Priority.YELLOW),
    Discriminator("fracture_closed", "Closed fracture (no break in skin)", Priority.YELLOW),
    Discriminator("burn_other", "Other burn", Priority.YELLOW),
    Discriminator("abdo_pain", "Abdominal pain", Priority.YELLOW),
    Discriminator("diabetic_gt17", "Diabetic glucose >17 (no ketonuria)", Priority.YELLOW),
    Discriminator("vomiting_persistent", "Persistent vomiting", Priority.YELLOW),
    Discriminator("pregnancy_trauma", "Pregnancy with trauma", Priority.YELLOW),
    Discriminator("pregnancy_pv_bleed", "Pregnancy with PV bleed", Priority.YELLOW),
    Discriminator("moderate_pain", "Moderate pain", Priority.YELLOW),
]

ALL_DISCRIMINATORS: list[Discriminator] = (
    EMERGENCY_SIGNS + VERY_URGENT_SIGNS + URGENT_SIGNS
)
DISCRIMINATORS_BY_CODE: dict[str, Discriminator] = {
    d.code: d for d in ALL_DISCRIMINATORS
}
