# South African Triage Scale (SATS) — Adult pathway (encoded reference)

> **Source / attribution.** This file is an original, fact-based re-encoding of the
> **Adult** triage criteria of the South African Triage Scale, for the purpose of a
> deterministic scoring engine. The thresholds below are clinical facts drawn from the
> *SATS Training Manual 2012* (Western Cape Government: Health / Emergency Medicine
> Society of South Africa). Charts and training text are **not** reproduced here.
>
> SATS materials are published by the **South African Triage Group (SATG)** / **EMSSA**
> under **CC BY-NC-SA 3.0**. This project ships only this re-encoded rule table (facts),
> cites the source, and does **not** redistribute the copyrighted SATS charts or manual.
> Credit: *South African Triage Group, SATS Training Manual 2012.*
>
> **This is a prioritisation aid, not a diagnosis.** A triage nurse or clinician confirms
> every priority level before a patient is moved.

Applies to patients **older than 12 years OR taller than 150 cm**. Younger/shorter
patients require the paediatric pathway (out of scope for this version).

---

## 1. Priority levels and target times

| Colour | Acuity | Target time to be seen | Disposition |
|--------|--------|------------------------|-------------|
| RED    | Emergency  | Immediate        | Resuscitation area |
| ORANGE | Very urgent | < 10 minutes    | Majors |
| YELLOW | Urgent     | < 1 hour         | Majors |
| GREEN  | Routine    | < 4 hours        | Non-urgent / designated area |
| BLUE   | Deceased   | < 2 hours        | Refer for certification |

*Source: SATS Training Manual 2012, "SATS priority levels and target times" (Table 3).*

## 2. Final priority rule (five-step approach)

1. **Look for emergency signs** and ask for the presenting complaint. Any emergency sign →
   **RED**, straight to resus — a TEWS does **not** need to be calculated first.
2. **Look for very urgent or urgent signs** → at least **ORANGE** / **YELLOW**.
3. **Measure the vital signs and calculate the TEWS** (section 4); map it to a band (section 3).
4. **Check the key additional investigations** (section 6) — they may change the priority.
5. **Assign the final priority = the highest acuity** implied by the signs and the TEWS band.
   The **senior healthcare professional (SHCP)** — clinical nurse practitioner or doctor — may
   override the final priority at their discretion. This is the mandatory human gate.

*Source: SATS Training Manual 2012, Section 2.1 "The five step approach" and the SATS process flowchart (Figure 2 / Figure 4).*

## 3. TEWS band → priority

| TEWS total | Band |
|------------|------|
| 0 – 2 | GREEN (routine) |
| 3 – 4 | YELLOW (urgent) |
| 5 – 6 | ORANGE (very urgent) |
| 7 or more | RED (emergency) |

*Source: SATS Training Manual 2012, Adult SATS process flowchart (Figure 4).*

## 4. Triage Early Warning Score (TEWS) — Adult

The TEWS is the sum of points for seven parameters. Each parameter scores 0–3.

| Parameter | +3 | +2 | +1 | 0 | +1 | +2 | +3 |
|-----------|----|----|----|---|----|----|----|
| Mobility | | | | Walking | With help | Stretcher / immobile | |
| Respiratory rate (breaths/min) | | < 9 | | 9–14 | 15–20 | 21–29 | > 29 |
| Heart rate (beats/min) | | < 41 | 41–50 | 51–100 | 101–110 | 111–129 | > 129 |
| Systolic BP (mmHg) | < 71 | 71–80 | 81–100 | 101–199 | | > 199 | |
| Temperature (°C) | | < 35 | | 35–38.4 | | > 38.4 | |
| AVPU | | Confused | | Alert | Reacts to voice | Reacts to pain | Unresponsive |
| Trauma | | | | No | Yes | | |

*Source: SATS Training Manual 2012, Adult SATS Chart "ADULT TEWS" table (Figure 4, p. 9).*
*Note: the printed chart places "Confused" and "Reacts to pain" both in the +2 column; "Reacts to voice" in +1; "Unresponsive" in +3.*

## 5. Adult discriminators

### 5.1 Emergency signs → RED (immediate)
- Obstructed airway / not breathing
- Current (active) seizure
- Facial or inhalation burn
- Hypoglycaemia (glucose 3 mmol/L or less)
- Cardiac arrest

### 5.2 Very urgent signs → ORANGE (< 10 min)
- Acute shortness of breath
- Reduced / confused level of consciousness
- Coughing blood
- Chest pain
- Stabbed neck
- Uncontrolled haemorrhage (arterial)
- Post-ictal (after seizure)
- Acute focal neurology (stroke)
- Aggression
- Threatened limb
- Eye injury
- Dislocation of a larger joint
- Compound fracture (break in skin)
- Burn >20% / electrical / circumferential / chemical
- Poisoning / overdose
- Diabetic glucose >11 with ketonuria
- Vomiting fresh blood
- Pregnancy with abdominal trauma or pain
- Severe pain
- High-energy transfer (severe mechanism)

"Reduced / confused level of consciousness" means **any patient who is not fully alert**
(confused, or responding only to voice or to pain, or unresponsive). It therefore follows
directly from the AVPU observation.

### 5.3 Urgent signs → YELLOW (< 1 hour)
- Controlled haemorrhage
- Dislocation of finger or toe
- Closed fracture (no break in skin)
- Other burn
- Abdominal pain
- Diabetic glucose >17 (no ketonuria)
- Persistent vomiting
- Pregnancy with trauma
- Pregnancy with PV bleed
- Moderate pain

**Pain.** The manual defines severe pain as unbearable (the worst the patient has ever felt)
and moderate pain as intense but bearable; mild pain is any other pain. *TriageFlow
convention (not from the manual):* a 0–10 pain score of 7–10 is offered as
severe and 4–6 as moderate, for the nurse to confirm.

*Source: SATS Training Manual 2012, Adult SATS Chart (Figure 4) and Section 3 "Adult clinical signs".*

## 6. Additional investigations (may change priority)

These are the adult chart's own prompts. The tool surfaces them as "do now" actions; it
performs none of them.

| When | Do |
|------|----|
| Respiratory rate scores 1 point or more on the TEWS | Check SpO₂ and hand over to the SHCP to give oxygen |
| Reduced level of consciousness (not alert, including confused) | Finger-prick glucose test and hand over to the SHCP |
| Unable to sit up / needs to lie down | Finger-prick glucose test and hand over to the SHCP |
| Active seizure / fitting, or recent seizure | Finger-prick glucose test and hand over to the SHCP |
| History of diabetes | Finger-prick glucose test |
| Hypoglycaemia — glucose 3 mmol/L or less | Move to resus and hand over to the SHCP (emergency, RED) |
| Diabetes and hyperglycaemia — glucose 11 mmol/L or more | Urine dipstick to check for ketones |
| Chest pain | Immediate ECG and hand over to the SHCP |
| Abdominal pain or backache in a female patient | Urine dipstick and urine pregnancy test |

*Source: SATS Training Manual 2012, Adult SATS Chart "Check for additional investigations" (Figure 4) and Section 6.*

## 7. Local conventions that are NOT part of SATS

Shown to the nurse labelled as local rules — never presented as SATS.

- **SpO₂ safety floor.** The SATS *adult* chart gives no SpO₂ threshold: it says to check
  SpO₂ and hand over to the SHCP for oxygen. (The < 92% → resus rule is on the *paediatric*
  chart.) As a conservative local rule, an adult SpO₂ below 92% is raised
  to at least ORANGE so the patient is seen within 10 minutes.
- **Deterioration watch.** A re-check that scores worse than the baseline raises a
  second-look flag for a clinician. It never moves the patient by itself.
- **Scope.** Patients younger than 12 years are not scored: the paediatric SATS chart applies.
