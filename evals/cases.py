"""Evaluation cases for TriageFlow Lite.

Four suites, written to probe behaviour honestly — under-triage, missing and impossible
data, the approval gate, and how well signs are found in free text — not curated to pass.

  * ``SCORING``     deterministic engine: is the SATS priority right?
  * ``FLOW``        the whole agent run: can an unsafe result reach the queue?
  * ``PARAPHRASES`` held-out lay phrasings of red-flag complaints (never used to build the
                    keyword lexicon) — measures sign-finding recall, keyword-only vs model.
  * ``BENIGN``      routine notes with pertinent negatives — nothing should be suggested.

``KNOWN_FAILURE`` is a real weakness we have NOT fixed (see EVALS.md).
"""

from triageflow.core.models import Encounter, VitalSigns
from triageflow.core.sats_rules import AVPU, Mobility

HEALTHY = dict(respiratory_rate=12, heart_rate=72, systolic_bp=122, temperature_c=36.7,
               avpu=AVPU.ALERT, mobility=Mobility.WALKING, trauma=False)


def enc(eid, complaint="", codes=None, age=45, **vit_over):
    v = {**HEALTHY, **vit_over}
    return Encounter(encounter_id=eid, age_years=age, presenting_complaint=complaint,
                     discriminator_codes=codes or [], vitals=VitalSigns(**v))


# (id, description, encounter, expectations, recheck_vitals or None)
SCORING = [
    ("E1", "Complete healthy adult → GREEN, fully scored",
     enc("E1", "routine BP check"),
     {"priority": "GREEN", "complete": True, "min_citations": 7}, None),
    ("E2", "Deranged vitals (septic-looking) → RED by TEWS",
     enc("E2", "fever and confusion", respiratory_rate=32, heart_rate=132,
         systolic_bp=88, temperature_c=39.6, avpu=AVPU.CONFUSED, mobility=Mobility.IMMOBILE),
     {"priority": "RED", "complete": True, "min_citations": 7}, None),
    ("E3", "Chest pain, normal vitals → ORANGE via sign (not TEWS) + ECG action",
     enc("E3", "central chest pain", codes=["chest_pain"]),
     {"priority": "ORANGE", "complete": True, "min_citations": 7, "action": "ecg"}, None),
    ("E4", "Emergency sign, no vitals at all → RED (TEWS not required)",
     Encounter(encounter_id="E4", age_years=60, presenting_complaint="collapsed, not breathing",
               discriminator_codes=["cardiac_arrest"], vitals=VitalSigns()),
     {"priority": "RED", "complete": True}, None),
    ("E5", "Missing required vitals, no overriding sign → INCOMPLETE (no band)",
     Encounter(encounter_id="E5", age_years=40, presenting_complaint="feels unwell",
               vitals=VitalSigns(heart_rate=88, systolic_bp=120)),
     {"priority": None, "complete": False, "missing_contains": "respiratory_rate"}, None),
    ("E6", "SpO2 88%, otherwise normal → ORANGE, labelled a LOCAL rule (not SATS)",
     enc("E6", "mild cough", spo2=88),
     {"priority": "ORANGE", "complete": True, "local_rule": True}, None),
    ("E7", "Urgent patient re-checked and worse → deterioration flag",
     enc("E7", "abdominal pain", codes=["abdo_pain"], respiratory_rate=22, heart_rate=104),
     {"priority": "YELLOW", "complete": True, "deterioration_escalates": True},
     dict(respiratory_rate=32, heart_rate=130, systolic_bp=90, temperature_c=37,
          avpu=AVPU.VOICE, mobility=Mobility.IMMOBILE, trauma=False, spo2=86)),
    ("E8", "Re-check clearly worse but temperature not retaken → flag still raised",
     enc("E8", "fever"),
     {"priority": "GREEN", "complete": True, "deterioration_escalates": True},
     dict(respiratory_rate=36, heart_rate=140, systolic_bp=75, avpu=AVPU.PAIN,
          mobility=Mobility.IMMOBILE, trauma=False)),
    ("E9", "Impossible HR 400 (entry error) → validation ERROR",
     enc("E9", "palpitations", heart_rate=400), {"validation_error": True}, None),
    ("E10", "Temperature entered in Fahrenheit → validation ERROR (unit catch)",
     enc("E10", "fever", temperature_c=101.2), {"validation_error": True}, None),
    ("E11", "Confused but vitals otherwise normal → ORANGE (SATS: not fully alert)",
     enc("E11", "family says not making sense", avpu=AVPU.CONFUSED),
     {"priority": "ORANGE", "complete": True, "action": "check_glucose"}, None),
    ("E12", "Responds only to voice, walking → ORANGE (was GREEN on TEWS alone)",
     enc("E12", "drowsy", avpu=AVPU.VOICE), {"priority": "ORANGE", "complete": True}, None),
    ("E13", "Glucose exactly 3.0 mmol/L → RED (chart: '3 or less')",
     enc("E13", "sweaty and shaky", glucose_mmol=3.0), {"priority": "RED", "complete": True}, None),
    ("E14", "Glucose 28 with no diabetic sign → not escalated, but ketone check prompted",
     enc("E14", "thirsty", glucose_mmol=28.0),
     {"priority": "GREEN", "complete": True, "action": "check_ketones"}, None),
    ("E15", "Breathing rate 3/min → scored (TEWS +2) with a warning, NOT rejected as a typo",
     enc("E15", "very drowsy after tablets", respiratory_rate=3),
     {"priority": "GREEN", "complete": True, "validation_warning": True, "action": "check_spo2"},
     None),
    ("E16", "4-year-old → out of scope, no adult band given",
     enc("E16", "fever", age=4, heart_rate=125, respiratory_rate=28),
     {"priority": None, "complete": False, "out_of_scope": True, "validation_error": True}, None),
    ("E17", "Decimal reading between bands (RR 14.5, HR 100.5) → scored, not 'missing'",
     enc("E17", "review", respiratory_rate=14.5, heart_rate=100.5),
     {"priority": "GREEN", "complete": True, "tews": 2}, None),
    ("E18", "Pain 8/10 → ORANGE; cut-off cited as a convention, nurse confirms",
     enc("E18", "back pain", pain_score=8), {"priority": "ORANGE", "complete": True}, None),
]

# (id, description, encounter dict, start kwargs, decision, expected outcome)
#   outcome: ("refused", "text in the reason") or ("confirmed", "COLOUR")
_OK = {"decision": "approve", "clinician_name": "Nurse A", "signs_checked": True}
_H = {k: (v.value if hasattr(v, "value") else v) for k, v in HEALTHY.items()}


def _e(eid, complaint="", age=45, **vit):
    return {"encounter_id": eid, "age_years": age, "presenting_complaint": complaint,
            "vitals": {**_H, **vit}}


FLOW = [
    ("F1", "Chest pain typed, normal vitals, sign NOT reviewed → approval refused",
     _e("F1", "central chest pain radiating to left arm"), {}, _OK,
     ("refused", "Review the suggested sign")),
    ("F2", "Same patient, nurse confirms chest pain → ORANGE confirmed",
     _e("F2", "central chest pain radiating to left arm"),
     {"nurse_confirmed_codes": ["chest_pain"]}, _OK, ("confirmed", "ORANGE")),
    ("F3", "Incomplete assessment → cannot be approved",
     {"encounter_id": "F3", "age_years": 40, "vitals": {"heart_rate": 88}}, {}, _OK,
     ("refused", "no priority to approve")),
    ("F4", "Override RED → GREEN with no reason → refused",
     _e("F4"), {"nurse_confirmed_codes": ["cardiac_arrest"]},
     {"decision": "override", "clinician_name": "Dr B", "override_priority": "GREEN", "reason": ""},
     ("refused", "written reason")),
    ("F5", "Mistyped vital (HR 1300) → no result at all, nothing to approve",
     _e("F5", heart_rate=1300), {}, _OK, ("refused", "not waiting for approval")),
    ("F6", "Emergency sign + mistyped vital → RED is not held up",
     _e("F6", heart_rate=1300), {"nurse_confirmed_codes": ["cardiac_arrest"]}, _OK,
     ("confirmed", "RED")),
    ("F7", "Model did not read the complaint, sign list not checked → refused",
     _e("F7", "feels like an elephant sitting on his chest"), {},
     {"decision": "approve", "clinician_name": "Nurse A"},
     ("refused", "did not read this complaint")),
    ("F8", "No approver name → refused",
     _e("F8"), {}, {"decision": "approve", "clinician_name": "", "signs_checked": True},
     ("refused", "name")),
]

# Held-out lay phrasings (written after, and never used to tune, the keyword lexicon).
PARAPHRASES = [
    ("tightness across the chest when climbing stairs, now at rest too", "chest_pain"),
    ("says his chest hurts and left arm feels heavy", "chest_pain"),
    ("pressure in the middle of the chest for 30 minutes", "chest_pain"),
    ("feels like an elephant sitting on his chest", "chest_pain"),
    ("struggling for air, cannot finish a sentence", "sob_acute"),
    ("breathing very fast and noisy since last night", "sob_acute"),
    ("out of breath walking to the toilet, worse today", "sob_acute"),
    ("asthmatic, chest tight and wheezy, inhaler not helping", "sob_acute"),
    ("one side of the face has dropped and speech is slurred", "focal_neuro_acute"),
    ("suddenly cannot lift the right arm, started 1 hour ago", "focal_neuro_acute"),
    ("woke up unable to move left leg and arm", "focal_neuro_acute"),
    ("drank insecticide about an hour ago", "poisoning_overdose"),
    ("swallowed many of her mother's blood pressure tablets", "poisoning_overdose"),
    ("child's father says he took too many sleeping pills", "poisoning_overdose"),
    ("knocked off his motorbike by a lorry", "high_energy_transfer"),
    ("fell from a mango tree, about 4 metres", "high_energy_transfer"),
    ("passenger in a matatu that overturned", "high_energy_transfer"),
    ("shaking all over and foaming at the mouth right now", "seizure_active"),
    ("had a fit this morning, now sleepy", "post_ictal"),
    ("brought in after fitting at school", "post_ictal"),
    ("bringing up blood when she coughs", "coughing_blood"),
    ("vomit looks like fresh red blood", "vomiting_fresh_blood"),
    ("deep cut on forearm, blood soaking through the towel", "haemorrhage_uncontrolled"),
    ("five months pregnant with bleeding from below", "pregnancy_pv_bleed"),
    ("expecting a baby, strong pains low in the tummy", "pregnancy_abdo"),
    ("very sleepy and hard to wake since morning", "loc_reduced"),
    ("family says she is talking nonsense and does not know where she is", "loc_reduced"),
    ("passed out twice today", "loc_reduced"),
    ("bone can be seen through the wound on his shin", "fracture_compound"),
    ("shoulder popped out playing rugby", "dislocation_large_joint"),
    ("acid splashed in the left eye", "eye_injury"),
    ("pain in the belly since yesterday, getting worse", "abdo_pain"),
    ("cannot stop being sick, nothing stays down", "vomiting_persistent"),
    ("fell on her arm, it looks bent, skin not broken", "fracture_closed"),
    ("boiling water spilled on his foot", "burn_other"),
    ("hit in the eye with a stone, vision blurred", "eye_injury"),
]

BENIGN = [
    "Here for repeat BP medication, feels well. No headache, no chest pain, no SOB.",
    "Sore throat and dry cough x3/7. Eating and drinking well. No difficulty breathing.",
    "For dressing change, L leg ulcer. Wound clean, no discharge, no fever.",
    "Known DM2 on metformin, here for refill. Feels well.",
    "omeprazole 20mg od, here for refill",
    "Wants a sick note for work, had a cold last week, better now.",
    "Itchy rash on both arms x1/52, no swelling of lips or tongue.",
    "Review of blood results, asymptomatic.",
    "Burning on passing urine x2/7. No fever, no flank pain, no vomiting.",
    "Family planning review, no complaints.",
]

# The weakness we have NOT fixed: with the language model off, the keyword matcher misses most
# lay phrasings, so a red-flag complaint can score GREEN with nothing suggested.
KNOWN_FAILURE = ("X1", "Model OFF + lay phrasing ('elephant sitting on his chest') → "
                       "nothing suggested, GREEN proposed",
                 _e("X1", "feels like an elephant sitting on his chest, sweating a lot"),
                 "ORANGE")
