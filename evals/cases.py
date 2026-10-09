"""Evaluation cases for TriageFlow Lite.

Each case is a realistic (synthetic) triage scenario with an expected outcome grounded in
the SATS Adult rules. The set is written to probe behaviour honestly — including under-
triage risk, missing data, bad units, overrides, and deterioration — not curated to pass.
Case E8 is a KNOWN FAILURE we have not fixed (see EVALS.md).
"""

from triageflow.core.models import Encounter, VitalSigns
from triageflow.core.sats_rules import AVPU, Mobility

HEALTHY = dict(respiratory_rate=12, heart_rate=72, systolic_bp=122, temperature_c=36.7,
               avpu=AVPU.ALERT, mobility=Mobility.WALKING, trauma=False)


def enc(eid, complaint="", codes=None, **vit_over):
    v = {**HEALTHY, **vit_over}
    return Encounter(encounter_id=eid, presenting_complaint=complaint,
                     discriminator_codes=codes or [], vitals=VitalSigns(**v))


# Each case: (id, description, encounter, expectations dict, recheck_vitals or None)
CASES = [
    ("E1", "Complete healthy adult → GREEN, fully scored",
     enc("E1", "routine BP check"),
     {"priority": "GREEN", "complete": True, "min_citations": 7}, None),

    ("E2", "Deranged vitals (septic-looking) → RED by TEWS",
     enc("E2", "fever and confusion", respiratory_rate=32, heart_rate=132,
         systolic_bp=88, temperature_c=39.6, avpu=AVPU.CONFUSED, mobility=Mobility.IMMOBILE),
     {"priority": "RED", "complete": True, "min_citations": 7}, None),

    ("E3", "Chest pain, normal vitals → ORANGE via discriminator (not TEWS)",
     enc("E3", "central chest pain", codes=["chest_pain"]),
     {"priority": "ORANGE", "complete": True, "min_citations": 7}, None),

    ("E4", "Emergency sign with incomplete vitals → RED, still complete (sign forces it)",
     Encounter(encounter_id="E4", presenting_complaint="collapsed, not breathing",
               discriminator_codes=["cardiac_arrest"], vitals=VitalSigns(heart_rate=40)),
     {"priority": "RED", "complete": True}, None),

    ("E5", "Missing required vitals, no overriding sign → INCOMPLETE (no band)",
     Encounter(encounter_id="E5", presenting_complaint="feels unwell",
               vitals=VitalSigns(heart_rate=88, systolic_bp=120)),
     {"priority": None, "complete": False, "missing_contains": "respiratory_rate"}, None),

    ("E6", "SpO2 92%-threshold flag raises an otherwise-green patient → ORANGE",
     enc("E6", "mild cough", spo2=88),
     {"priority": "ORANGE", "complete": True}, None),

    ("E7", "Deterioration on re-check: ORANGE baseline escalates (second-look flag)",
     enc("E7", "abdominal pain", codes=["abdo_pain"], respiratory_rate=22, heart_rate=104),
     {"priority": "YELLOW", "complete": True, "deterioration_escalates": True},
     dict(respiratory_rate=32, heart_rate=130, systolic_bp=90, temperature_c=37,
          avpu=AVPU.VOICE, mobility=Mobility.IMMOBILE, trauma=False, spo2=86)),

    # --- KNOWN FAILURE (unfixed) ---
    ("E8", "Free-text-only emergency, no structured sign, keyword miss → UNDER-TRIAGE",
     enc("E8", "crushing retrosternal discomfort radiating to jaw"),  # no 'chest pain' tokens, no code
     # Clinically correct = ORANGE (chest pain). System outputs GREEN → documented failure.
     {"priority": "ORANGE", "complete": True, "note": "KNOWN FAILURE: system under-triages to GREEN"}, None),

    ("E9", "Implausible HR (entry error) → validation ERROR",
     enc("E9", "palpitations", heart_rate=400),
     {"validation_error": True}, None),

    ("E10", "Temperature entered in Fahrenheit → validation ERROR (unit catch)",
     enc("E10", "fever", temperature_c=101.2),
     {"validation_error": True}, None),
]
