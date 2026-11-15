"""Synthetic adult triage patients — reproducible, EMR-shaped, no real patient data.

Each patient is drawn from a **scenario**: a short clinical vignette written the way a triage
nurse actually notes it (C/O = complains of, Hx = history, x3/7 = for 3 days, RBS = random
blood sugar, PV = per vagina, RTA = road traffic accident), together with vitals that are
**coherent with that story**. A patient in cardiac arrest has no recorded vitals; an
asthmatic is breathing fast with a low SpO2; someone here for a BP refill is walking, alert
and pain-free. Numbers quoted in the note (pain score, glucose) are the same numbers in the
structured vitals.

Vitals are sampled inside ranges chosen so that every scenario lands on a known SATS colour
once the nurse confirms its signs (``Scenario.expected``) — pinned by the test-suite for
many seeds — so the data can be used both for demos and as an evaluation set.

Signs are stored as ground truth (``Encounter.discriminator_codes``) for evaluation, but the
UI never pre-applies them: the nurse confirms signs, exactly as with a real patient.
Everything is seedable so runs are reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from ..core.models import Encounter, VitalSigns
from ..core.sats_rules import AVPU, Mobility

Range = tuple[float, float]


@dataclass(frozen=True)
class Scenario:
    key: str
    profile: str                 # acuity bucket: stable | moderate | severe | critical
    expected: str                # SATS colour once the signs below are confirmed
    complaint: str               # triage note; may use {pain} and {glucose}
    codes: tuple[str, ...]       # SATS signs the nurse would confirm (ground truth)
    age: tuple[int, int]
    rr: Range
    hr: Range
    sbp: Range
    temp: Range
    spo2: Range
    pain: tuple[int, int]
    sex: str | None = None       # None = either
    avpu: AVPU = AVPU.ALERT
    mobility: Mobility = Mobility.WALKING
    trauma: bool = False
    glucose: Range | None = None
    vitals_taken: bool = True    # False: straight to resus, nothing measured at the desk


_N = dict(rr=(12, 14), hr=(62, 92), sbp=(108, 148), temp=(36.2, 37.3), spo2=(96, 99))  # normal

SCENARIOS: list[Scenario] = [
    # ---- GREEN / routine ----------------------------------------------------------------
    Scenario("bp_refill", "stable", "GREEN",
             "Here for repeat BP medication, feels well. No headache, no chest pain, no SOB. "
             "Last seen 3/12 ago.",
             (), (40, 78), **{**_N, "sbp": (128, 162)}, pain=(0, 0)),
    Scenario("sore_throat", "stable", "GREEN",
             "Sore throat and dry cough x3/7. Eating and drinking well. No fever at home, "
             "no difficulty breathing.",
             (), (16, 45), **{**_N, "temp": (36.6, 37.6)}, pain=(1, 3)),
    Scenario("ankle_sprain", "stable", "GREEN",
             "Twisted R ankle playing football yesterday. Swelling++, weight-bearing, "
             "no deformity. Pain {pain}/10.",
             (), (16, 40), **_N, pain=(2, 3), trauma=True),
    Scenario("dressing", "stable", "GREEN",
             "For dressing change, L leg ulcer. Wound clean, no discharge, no fever. "
             "Mobilising independently.",
             (), (45, 80), **_N, pain=(0, 2)),
    Scenario("dm_refill", "stable", "GREEN",
             "Known DM2 on metformin, here for refill and RBS check. Feels well, "
             "no polyuria. RBS {glucose}.",
             (), (40, 75), **_N, pain=(0, 0), glucose=(5.8, 9.6)),
    Scenario("dysuria", "stable", "GREEN",
             "Burning on passing urine x2/7, frequency. No fever, no flank pain, no vomiting.",
             (), (18, 50), **_N, pain=(2, 3), sex="female"),

    # ---- YELLOW / urgent ----------------------------------------------------------------
    Scenario("abdo_pain", "moderate", "YELLOW",
             "C/O abdominal pain since this morning, periumbilical, nausea x1. No vomiting, "
             "no diarrhoea. LMP 2/52 ago. Pain {pain}/10.",
             ("abdo_pain",), (18, 40), rr=(15, 18), hr=(84, 100), sbp=(108, 135),
             temp=(36.8, 37.8), spo2=(97, 99), pain=(4, 6), sex="female"),
    Scenario("wrist_fracture", "moderate", "YELLOW",
             "Fell onto outstretched R hand ~2 hrs ago. Wrist swollen and deformed, ?fracture. "
             "Skin intact, fingers warm and moving. Pain {pain}/10.",
             ("fracture_closed",), (18, 70), rr=(15, 18), hr=(84, 100), sbp=(112, 145),
             temp=(36.3, 37.2), spo2=(97, 99), pain=(5, 6), trauma=True),
    Scenario("vomiting", "moderate", "YELLOW",
             "Vomiting x6 since last night, not keeping fluids down. No blood in vomit. "
             "Mild cramps, passing urine.",
             ("vomiting_persistent",), (18, 60), rr=(15, 18), hr=(92, 100), sbp=(102, 125),
             temp=(36.8, 37.7), spo2=(97, 99), pain=(2, 3)),
    Scenario("fever", "moderate", "YELLOW",
             "Fever, chills and body aches x3/7 with headache. No neck stiffness, "
             "no confusion, drinking fluids. Malaria RDT pending.",
             (), (16, 55), rr=(16, 20), hr=(102, 110), sbp=(104, 128),
             temp=(38.6, 39.6), spo2=(96, 99), pain=(2, 3)),
    Scenario("scald", "moderate", "YELLOW",
             "Hot water scald to L forearm 1 hr ago, about palm-sized, blistering. "
             "Hand moving normally. Pain {pain}/10.",
             ("burn_other",), (16, 60), rr=(15, 18), hr=(84, 100), sbp=(110, 140),
             temp=(36.3, 37.2), spo2=(97, 99), pain=(4, 6), trauma=True),
    Scenario("hyperglycaemia", "moderate", "YELLOW",
             "Known diabetic, tired and very thirsty, passing a lot of urine x2/7. "
             "RBS {glucose}, urine dipstick: no ketones.",
             ("diabetic_gt17",), (35, 75), rr=(14, 18), hr=(84, 100), sbp=(108, 140),
             temp=(36.4, 37.3), spo2=(96, 99), pain=(0, 2), glucose=(17.6, 24.5)),
    Scenario("pv_spotting", "moderate", "YELLOW",
             "G2P1, about 12/40 pregnant. PV spotting since this morning. "
             "No abdominal pain, no dizziness.",
             ("pregnancy_pv_bleed",), (18, 38), rr=(14, 18), hr=(80, 100), sbp=(102, 128),
             temp=(36.4, 37.3), spo2=(97, 99), pain=(0, 2), sex="female"),

    # ---- ORANGE / very urgent -----------------------------------------------------------
    Scenario("chest_pain", "severe", "ORANGE",
             "C/O central chest pain x1 hr, heavy and pressing, radiating to L arm. "
             "Sweating, nausea. Hx HTN, DM2. Pain {pain}/10.",
             ("chest_pain",), (45, 78), rr=(18, 24), hr=(92, 110), sbp=(108, 168),
             temp=(36.2, 37.2), spo2=(94, 98), pain=(7, 9), mobility=Mobility.WITH_HELP),
    Scenario("asthma", "severe", "ORANGE",
             "SOB since last night, worse this morning. Speaking in short sentences, "
             "audible wheeze. Known asthmatic, inhaler x4 with little relief.",
             ("sob_acute",), (16, 55), rr=(24, 29), hr=(104, 124), sbp=(108, 140),
             temp=(36.4, 37.4), spo2=(89, 93), pain=(0, 2), mobility=Mobility.WITH_HELP),
    Scenario("stroke", "severe", "ORANGE",
             "Sudden weakness of R arm and leg with facial droop and slurred speech, "
             "noticed ~45 min ago by family. Hx HTN.",
             ("focal_neuro_acute",), (50, 85), rr=(14, 18), hr=(70, 98), sbp=(162, 198),
             temp=(36.2, 37.2), spo2=(95, 98), pain=(0, 1), mobility=Mobility.IMMOBILE),
    Scenario("pregnant_abdo", "severe", "ORANGE",
             "G3P2, about 30/40 pregnant. Severe lower abdominal pain x2 hrs, feels faint. "
             "No PV bleeding noted. Pain {pain}/10.",
             ("pregnancy_abdo",), (18, 40), rr=(16, 20), hr=(96, 110), sbp=(101, 122),
             temp=(36.4, 37.3), spo2=(96, 99), pain=(7, 9), sex="female",
             mobility=Mobility.WITH_HELP),
    Scenario("overdose", "severe", "ORANGE",
             "Brought by sister — took about 20 paracetamol tablets 3 hrs ago after an "
             "argument. Awake, tearful, nauseated.",
             ("poisoning_overdose",), (16, 40), rr=(14, 18), hr=(84, 104), sbp=(106, 130),
             temp=(36.3, 37.2), spo2=(97, 99), pain=(0, 2)),
    Scenario("rta", "severe", "ORANGE",
             "RTA — boda boda passenger hit by a car ~30 min ago, thrown off. "
             "C/O R thigh pain with deformity. Did not lose consciousness. Pain {pain}/10.",
             ("high_energy_transfer",), (18, 50), rr=(16, 20), hr=(96, 110), sbp=(105, 128),
             temp=(36.0, 37.0), spo2=(95, 99), pain=(7, 9), trauma=True,
             mobility=Mobility.IMMOBILE),
    Scenario("post_ictal", "severe", "ORANGE",
             "Had a seizure at home ~20 min ago lasting about 3 min per wife. Now drowsy but "
             "rousable. Known epileptic, missed doses x2/7.",
             ("post_ictal",), (18, 60), rr=(14, 20), hr=(88, 108), sbp=(110, 140),
             temp=(36.4, 37.5), spo2=(94, 98), pain=(0, 2), avpu=AVPU.VOICE,
             mobility=Mobility.WITH_HELP),
    Scenario("haematemesis", "severe", "ORANGE",
             "Vomited fresh blood x2 this morning, about a cupful each time. "
             "Dizzy on standing. Hx alcohol use.",
             ("vomiting_fresh_blood",), (30, 70), rr=(16, 20), hr=(98, 110), sbp=(101, 115),
             temp=(36.2, 37.2), spo2=(95, 98), pain=(2, 3), mobility=Mobility.WITH_HELP),

    # ---- RED / emergency ----------------------------------------------------------------
    Scenario("cardiac_arrest", "critical", "RED",
             "Found collapsed at home, brought in by taxi. Not breathing, no pulse felt — "
             "CPR started at triage.",
             ("cardiac_arrest",), (45, 85), **_N, pain=(0, 0), avpu=AVPU.UNRESPONSIVE,
             mobility=Mobility.IMMOBILE, vitals_taken=False),
    Scenario("seizure", "critical", "RED",
             "Convulsing on arrival, ongoing about 5 min per relatives. "
             "Not responding to voice. Known epileptic.",
             ("seizure_active",), (16, 55), rr=(22, 30), hr=(118, 140), sbp=(128, 160),
             temp=(37.0, 38.0), spo2=(85, 92), pain=(0, 0), avpu=AVPU.PAIN,
             mobility=Mobility.IMMOBILE),
    Scenario("inhalation_burn", "critical", "RED",
             "Rescued from a house fire ~40 min ago. Burns to face and neck, soot around "
             "mouth and nose, hoarse voice, coughing.",
             ("burn_face_inhalation",), (16, 65), rr=(24, 30), hr=(110, 128), sbp=(105, 140),
             temp=(36.4, 37.4), spo2=(88, 94), pain=(7, 9), trauma=True,
             mobility=Mobility.WITH_HELP),
    Scenario("hypoglycaemia", "critical", "RED",
             "Known diabetic on insulin, found confused and sweating by family. "
             "Not eaten since morning. RBS {glucose}.",
             ("hypoglycaemia_lt3",), (30, 80), rr=(16, 20), hr=(96, 116), sbp=(108, 140),
             temp=(35.8, 36.8), spo2=(95, 99), pain=(0, 0), avpu=AVPU.CONFUSED,
             mobility=Mobility.WITH_HELP, glucose=(1.6, 2.9)),
    Scenario("sepsis", "critical", "RED",
             "Fever and rigors x2/7, productive cough. Now confused and too weak to stand, "
             "breathing fast.",
             (), (35, 85), rr=(30, 36), hr=(130, 146), sbp=(78, 96), temp=(39.0, 40.2),
             spo2=(84, 91), pain=(0, 2), avpu=AVPU.CONFUSED, mobility=Mobility.IMMOBILE),
]

SCENARIOS_BY_KEY = {s.key: s for s in SCENARIOS}
_BY_PROFILE: dict[str, list[Scenario]] = {}
for _s in SCENARIOS:
    _BY_PROFILE.setdefault(_s.profile, []).append(_s)
PROFILES = tuple(_BY_PROFILE)   # ("stable", "moderate", "severe", "critical")


def _u(rng: random.Random, r: Range) -> float:
    return rng.uniform(*r)


def make_scenario(key: str, seed: int, include_spo2: bool = True) -> Encounter:
    """One synthetic patient for a named scenario (reproducible for a given seed)."""
    s = SCENARIOS_BY_KEY[key]
    rng = random.Random(f"{key}:{seed}")
    arrival = datetime.now(UTC) - timedelta(minutes=rng.randint(0, 90))
    pain = rng.randint(*s.pain)
    glucose = round(_u(rng, s.glucose), 1) if s.glucose else None
    if s.vitals_taken:
        systolic = round(_u(rng, s.sbp))
        diastolic = round(min(max(systolic * 0.63 + rng.gauss(0, 4), 40), systolic - 15))
        vitals = VitalSigns(
            respiratory_rate=round(_u(rng, s.rr)), heart_rate=round(_u(rng, s.hr)),
            systolic_bp=systolic, diastolic_bp=diastolic,
            temperature_c=round(_u(rng, s.temp), 1),
            avpu=s.avpu, mobility=s.mobility, trauma=s.trauma,
            spo2=round(_u(rng, s.spo2)) if include_spo2 else None,
            glucose_mmol=glucose, weight_kg=round(min(max(rng.gauss(70, 14), 42), 125), 1),
            pain_score=pain, measured_at=arrival + timedelta(minutes=rng.randint(1, 8)),
        )
    else:   # straight to resus: only what can be seen at a glance is recorded
        vitals = VitalSigns(avpu=s.avpu, mobility=s.mobility, trauma=s.trauma,
                            measured_at=arrival + timedelta(minutes=1))
    return Encounter(
        encounter_id=f"SYN-{s.profile[:3].upper()}-{seed % 100000:05d}",
        age_years=rng.randint(*s.age), sex=s.sex or rng.choice(["male", "female"]),
        presenting_complaint=s.complaint.format(
            pain=pain, glucose=f"{glucose:.1f}" if glucose is not None else ""),
        arrival_time=arrival, vitals=vitals, discriminator_codes=list(s.codes),
    )


def make_encounter(profile_name: str, seed: int, include_spo2: bool = True) -> Encounter:
    """One synthetic patient of the given acuity bucket (stable/moderate/severe/critical)."""
    rng = random.Random(f"{profile_name}:{seed}")
    scenario = rng.choice(_BY_PROFILE[profile_name])
    return make_scenario(scenario.key, seed, include_spo2=include_spo2)


def stream(n: int = 12, seed: int = 42,
           mix=("stable", "stable", "moderate", "moderate", "severe", "critical")):
    """A reproducible queue of mixed-acuity encounters (no scenario repeated until all of
    that acuity have been used)."""
    rng = random.Random(seed)
    pools = {p: rng.sample(v, len(v)) for p, v in _BY_PROFILE.items()}
    used = dict.fromkeys(pools, 0)
    out = []
    for i in range(n):
        profile = mix[i % len(mix)]
        scenario = pools[profile][used[profile] % len(pools[profile])]
        used[profile] += 1
        out.append(make_scenario(scenario.key, seed=seed * 1000 + i))
    rng.shuffle(out)
    return out


def deterioration_series(seed: int = 7):
    """A patient who looks urgent-but-stable at triage (fever, TEWS 4), then worsens on
    re-check 35 minutes later — for the deterioration watch."""
    base = make_scenario("fever", seed)
    base.encounter_id = f"SYN-DET-{seed:04d}"
    recheck = base.model_copy(deep=True)
    recheck.vitals = base.vitals.model_copy(update={
        "respiratory_rate": base.vitals.respiratory_rate + 10,
        "heart_rate": base.vitals.heart_rate + 25,
        "systolic_bp": base.vitals.systolic_bp - 22,
        "spo2": (base.vitals.spo2 or 95) - 8,
        "avpu": AVPU.VOICE,
        "measured_at": base.vitals.measured_at + timedelta(minutes=35),
    })
    return base, recheck
