"""Synthetic adult triage patient stream — reproducible, EMR-shaped, no real patient data.

Treat synthetic as the default (challenge guidance). We generate plausible vitals by
sampling around physiological norms and, for higher-acuity cases, perturbing specific
parameters. Output can be rendered to FHIR via ``data.fhir``.

Everything is seedable so runs are reproducible and evals are stable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from ..core.models import Encounter, VitalSigns
from ..core.sats_rules import AVPU, Mobility

# Presenting complaints written as realistic triage-desk notes: the patient's symptoms in
# nurse shorthand (C/O = complains of, Hx = history, N = nausea, PV = per vagina, x3/7 =
# for 3 days), with onset, associated symptoms and pertinent negatives — never a diagnosis.
# Terse for low-acuity, richer for urgent. The free text is deliberately messy so the agent's
# extraction does real work; the structured discriminator codes are the confirmed truth.
COMPLAINTS_GREEN = [
    ("Here for repeat BP meds, feels well, no complaints today", []),
    ("Sore throat x3/7, eating & drinking OK, no fever at home", []),
    ("Twisted R ankle yest, swelling+ but weight-bearing, no deformity", []),
    ("Here for BP check, asymptomatic", []),
]
COMPLAINTS_YELLOW = [
    ("C/O abdo pain since AM, periumbilical, N x1, no vomiting/diarrhoea. Hx nil", ["abdo_pain"]),
    ("Fell onto R wrist ~2hrs ago, swollen+deformed, ?fracture, fingers mobile", ["fracture_closed"]),
    ("Persistent vomiting x6 since last night, tolerating no fluids, no blood seen", ["vomiting_persistent"]),
    ("C/O lower back pain x2/7, moderate pain, worse on movement, no leg weakness", ["moderate_pain"]),
]
COMPLAINTS_ORANGE = [
    ("C/O central chest pain x1hr, pressure-like, radiates to L arm, diaphoretic. Hx HTN, DM2", ["chest_pain"]),
    ("SOB since last night, worse lying flat, speaking in short sentences. Hx asthma", ["sob_acute"]),
    ("Pregnant ~30/40 c/o severe abdo pain + feels faint, no PV bleed noted", ["pregnancy_abdo", "severe_pain"]),
    ("Facial droop + slurred speech since ~45min per family, R arm weak, ?stroke", ["focal_neuro_acute"]),
]
COMPLAINTS_RED = [
    ("Found collapsed at home per EMS, not breathing on arrival, CPR ongoing", ["cardiac_arrest"]),
    ("Witnessed seizure on arrival, still convulsing, not responding to voice", ["seizure_active"]),
    ("Burns to face from house fire, soot around mouth, hoarse voice", ["burn_face_inhalation"]),
]


@dataclass
class Profile:
    name: str
    rr: tuple[float, float]
    hr: tuple[float, float]
    sbp: tuple[float, float]
    temp: tuple[float, float]
    spo2: tuple[float, float]
    avpu_weights: dict[AVPU, float]
    mobility_weights: dict[Mobility, float]


PROFILES = {
    "stable": Profile("stable", (14, 2), (75, 8), (122, 10), (36.8, 0.3), (98, 1),
                      {AVPU.ALERT: 1.0},
                      {Mobility.WALKING: 0.9, Mobility.WITH_HELP: 0.1}),
    "moderate": Profile("moderate", (22, 3), (105, 10), (108, 12), (38.0, 0.6), (95, 2),
                        {AVPU.ALERT: 0.8, AVPU.VOICE: 0.2},
                        {Mobility.WALKING: 0.5, Mobility.WITH_HELP: 0.5}),
    "severe": Profile("severe", (28, 4), (124, 12), (92, 14), (38.8, 0.8), (90, 3),
                      {AVPU.ALERT: 0.4, AVPU.VOICE: 0.4, AVPU.CONFUSED: 0.2},
                      {Mobility.WITH_HELP: 0.5, Mobility.IMMOBILE: 0.5}),
    "critical": Profile("critical", (34, 6), (135, 18), (82, 18), (39.2, 1.0), (86, 4),
                        {AVPU.VOICE: 0.3, AVPU.PAIN: 0.4, AVPU.UNRESPONSIVE: 0.3},
                        {Mobility.IMMOBILE: 1.0}),
}

_LEVELS = {
    "stable": COMPLAINTS_GREEN,
    "moderate": COMPLAINTS_YELLOW,
    "severe": COMPLAINTS_ORANGE,
    "critical": COMPLAINTS_RED,
}


def _weighted(rng: random.Random, weights: dict):
    items, w = zip(*weights.items(), strict=True)
    return rng.choices(items, weights=w, k=1)[0]


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def make_encounter(profile_name: str, seed: int, include_spo2: bool = True) -> Encounter:
    rng = random.Random(seed)
    p = PROFILES[profile_name]
    complaint, codes = rng.choice(_LEVELS[profile_name])
    arrival = datetime.now(UTC) - timedelta(minutes=rng.randint(0, 90))
    systolic = round(_clip(rng.gauss(*p.sbp), 60, 210))
    # Diastolic ~ 0.63 x systolic with a little noise (kept below systolic); pain by acuity.
    diastolic = round(_clip(systolic * 0.63 + rng.gauss(0, 5), 30, systolic - 10))
    pain_by_profile = {"stable": (1, 1), "moderate": (5, 1), "severe": (7, 1.5), "critical": (8, 1.5)}
    pain = int(_clip(round(rng.gauss(*pain_by_profile[profile_name])), 0, 10))
    vitals = VitalSigns(
        respiratory_rate=round(_clip(rng.gauss(*p.rr), 6, 48)),
        heart_rate=round(_clip(rng.gauss(*p.hr), 35, 180)),
        systolic_bp=systolic,
        diastolic_bp=diastolic,
        temperature_c=round(_clip(rng.gauss(*p.temp), 34.0, 41.0), 1),
        avpu=_weighted(rng, p.avpu_weights),
        mobility=_weighted(rng, p.mobility_weights),
        trauma=rng.random() < 0.25,
        spo2=round(_clip(rng.gauss(*p.spo2), 70, 100)) if include_spo2 else None,
        weight_kg=round(_clip(rng.gauss(70, 15), 40, 130), 1),
        pain_score=pain,
        measured_at=arrival + timedelta(minutes=rng.randint(1, 8)),
    )
    return Encounter(
        encounter_id=f"SYN-{profile_name[:3].upper()}-{seed:04d}",
        age_years=rng.randint(16, 85), sex=rng.choice(["male", "female"]),
        presenting_complaint=complaint, arrival_time=arrival,
        vitals=vitals, discriminator_codes=codes,
    )


def stream(n: int = 12, seed: int = 42,
           mix=("stable", "stable", "moderate", "moderate", "severe", "critical")):
    """A reproducible queue of mixed-acuity encounters."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        profile = mix[i % len(mix)]
        out.append(make_encounter(profile, seed=seed * 1000 + i))
    rng.shuffle(out)
    return out


def deterioration_series(seed: int = 7):
    """A patient who looks moderate at triage, then worsens on re-check (for deterioration watch)."""
    base = make_encounter("moderate", seed)
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
