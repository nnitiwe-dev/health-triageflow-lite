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

1. **Emergency signs** present → **RED**, go straight to resus (no TEWS needed).
2. Else check **very urgent** signs → at least **ORANGE**.
3. Else check **urgent** signs → at least **YELLOW**.
4. Measure vitals and compute the **TEWS** (section 4); map the score to a band (section 3).
5. **Final priority = the highest acuity** implied by the discriminator signs and the TEWS band.
   A **Senior Health Care Professional (SHCP)** may override the computed level at their
   discretion — this is the mandatory human gate.

*Source: SATS Training Manual 2012, "The five step approach" and the Adult SATS process flowchart (Figure 2 / Figure 4).*

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
- Hypoglycaemia — glucose less than 3 mmol/L
- Cardiac arrest

### 5.2 Very urgent signs → ORANGE (< 10 min)
- Acute shortness of breath
- Reduced / confused level of consciousness
- Coughing blood
- Chest pain
- Stabbed neck
- Uncontrolled haemorrhage (arterial bleed)
- Post-ictal (after a seizure)
- Acute focal neurology (stroke)
- Aggression
- Threatened limb
- Eye injury
- Dislocation of a larger joint (not finger or toe)
- Compound fracture (break in the skin)
- Burn over 20%; electrical, circumferential, or chemical burn
- Poisoning / overdose
- Diabetic — glucose over 11 mmol/L with ketonuria
- Vomiting fresh blood
- Pregnancy with abdominal trauma or abdominal pain
- Severe pain
- High-energy transfer (severe mechanism of injury)

### 5.3 Urgent signs → YELLOW (< 1 hour)
- Controlled haemorrhage
- Dislocation of a finger or toe
- Closed fracture (no break in skin)
- Other burn
- Abdominal pain
- Diabetic — glucose over 17 mmol/L (no ketonuria)
- Persistent vomiting
- Pregnancy with trauma
- Pregnancy with PV (vaginal) bleed
- Moderate pain

## 6. Additional investigations (may change priority)
- If RR scores ≥ 1 on TEWS → check **SpO₂** and hand to SHCP for oxygen. SpO₂ < 92% is a red flag.
- Diabetic / altered consciousness → finger-prick **glucose**; < 3 mmol/L → emergency.
- Chest pain → immediate ECG and hand to SHCP.

*Source: SATS Training Manual 2012, "Check for additional investigations" (Adult SATS Chart) and Section 6.*
