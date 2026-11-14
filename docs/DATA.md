# Data card — synthetic triage data (for stakeholders)

**Audience:** triage nurses, clinical leads, facility IT, data-governance reviewers, and
judges. **Purpose:** explain exactly what data TriageFlow Lite uses, how it is produced, why
you can trust it, and where it must *not* be trusted.

> **One sentence:** TriageFlow Lite ships and runs on **synthetic, EMR-shaped data only** —
> there is no real patient data anywhere in this project, and patient data entered into a
> deployment never leaves the machine.

---

## 1. Why synthetic (the governance stance)

Patient health data is sensitive regardless of source. Our default is synthetic data, not as
a fallback but as the design choice:

- **No real patient data** is included, downloaded, or required — not even de-identified sets
  (e.g. MIMIC), which remain real-person data with their own residency and consent concerns.
- **No restricted ontologies.** We use **LOINC** (free) for vital-sign codes and a small
  self-authored code system for complaints. We deliberately avoid **SNOMED CT**, whose
  licence requires Affiliate status and carries fees in non-member countries.
- **Residency.** Inference runs locally (llama.cpp or Ollama); the configured data region is recorded in
  config and shown in the UI header. Nothing is sent to an external service.

## 2. What is generated

Adult (older than 12 years / taller than 150 cm) triage encounters, each with:

| Part | Fields |
|------|--------|
| Demographics | age, sex (synthetic, constrained to fit the story) |
| Presenting complaint | a triage note in nurse shorthand + the SATS signs a nurse would confirm |
| Vitals | breathing rate, pulse rate, systolic + diastolic BP, temperature, AVPU, mobility, trauma |
| Additional | SpO₂, finger-prick glucose (where the story calls for it), pain 0–10, weight |
| Timing | arrival time, measurement time |

Produced as:
- **Native FHIR R4** resources (`Patient`, `Encounter`, `Condition`, `Observation`) so an
  encounter looks like what a real EMR/FHIR API returns (`src/triageflow/data/fhir.py`).
- A reproducible **waiting queue** and a **deterioration series** (baseline → worsened
  re-check) for the deterioration-watch demo (`src/triageflow/data/synthetic.py`).

## 3. How it is generated (method & explainability)

Every patient comes from one of **26 scenarios** — a short clinical vignette plus the vitals
that go with it. Six are routine (GREEN), seven urgent (YELLOW), eight very urgent (ORANGE)
and five emergencies (RED).

1. **The note** is written the way a triage nurse writes it: onset, what the patient reports,
   pertinent negatives, relevant history, in common shorthand (C/O, Hx, x3/7, RBS, PV, RTA,
   G2P1, 30/40). For example:
   - *"Here for repeat BP medication, feels well. No headache, no chest pain, no SOB. Last
     seen 3/12 ago."*
   - *"Fell onto outstretched R hand ~2 hrs ago. Wrist swollen and deformed, ?fracture. Skin
     intact, fingers warm and moving. Pain 6/10."*
   - *"Known diabetic on insulin, found confused and sweating by family. Not eaten since
     morning. RBS 2.1."*
   - *"RTA — boda boda passenger hit by a car ~30 min ago, thrown off. C/O R thigh pain with
     deformity. Did not lose consciousness. Pain 8/10."*
2. **The vitals fit the story.** Each scenario has its own range for every vital, sampled
   uniformly: the asthmatic breathes at 24–29/min with SpO₂ 89–93%; the patient here for a
   refill is walking, alert and pain-free; the stroke patient is hypertensive and cannot
   walk; **the patient in cardiac arrest has no desk vitals at all**. Diastolic BP follows
   systolic; trauma, mobility and alertness are set by the scenario, not rolled at random.
3. **Numbers in the note are the numbers in the record** — the pain score and glucose quoted
   in the text are the generated values.
4. **Demographics fit**: pregnancy scenarios are women of child-bearing age; the stroke and
   chest-pain patients are older adults.

**Is the answer leaked?** Each scenario states the SATS colour it is *designed* to reach, but
the data carries no label that the engine reads: the priority is always computed from the
vitals and the signs the nurse confirms. The designed colour is used only to *check* the
engine — `tests/test_synthetic.py` and the eval suite score every scenario for many seeds and
fail if any lands elsewhere. In the UI the signs are **not** pre-applied: a loaded synthetic
patient has to have its signs confirmed, exactly like a real one.

**Reproducibility.** Everything is **seeded** (`stream(seed=…)`, `make_scenario(key, seed)`,
`make_encounter(profile, seed)`). The same seed yields the same patient on any machine.

## 4. Why you can trust it (and audit it)

- **Transparent & inspectable.** The scenarios are a readable table in
  `src/triageflow/data/synthetic.py` — the note, the signs and the range of every vital are
  written out; there is no opaque model producing the data.
- **Checked three ways** by the test-suite, for every scenario and many seeds: it scores to
  its intended colour; its vitals raise no validation error or "extreme value" warning; and
  the keyword matcher alone surfaces every sign its note describes, while the routine notes
  (full of "no chest pain, no SOB") suggest nothing.
- **Deterministic pipeline.** Data in → the deterministic SATS engine → a priority with a
  citation for every point. Given the same encounter, the same priority, every time.
- **FHIR-validatable.** Bundles can be round-tripped (`vitals_from_observations`) and fed to a
  real FHIR server (interoperability extension in ARCHITECTURE.md) for structural validation.
- **Audit trail.** Every assessment (tool calls, inputs/outputs, timestamps, and the named
  clinician approval) is written append-only to `~/.triageflow/audit.jsonl` and
  `~/.triageflow/triageflow.db`.

## 5. Limitations & non-use (important)

- **Written for face validity, not yet reviewed at the bedside.** The vignettes follow common
  triage shorthand and the SATS chart, but they were authored by the project team. They
  should be read and corrected by practising triage nurses in the target country before
  being used for training or a pilot — wording, abbreviations and case mix differ by site.
- **Not epidemiology.** The mix of scenarios exercises every SATS pathway; it does not
  reflect how often each presentation arrives at a real OPD.
- **Ranges are tidy.** Vitals are sampled inside bands chosen so each scenario reaches a known
  colour. Real patients are messier: borderline, contradictory and incomplete readings are
  covered by the eval cases, not by the scenario generator.
- **Not a validation set.** Passing evals on this data shows the *engine behaves as
  specified*; it does **not** establish clinical safety or accuracy on real patients.
- **Adult only, English notes.** Paediatric physiology and other languages are out of scope.
- **Do not substitute for consented real data** when moving toward a pilot. A facility pilot
  must use the locally-adopted protocol and data governed by that facility's consent/ethics
  process — never real patient data without consent, even de-identified.

## 6. Reproduce / inspect it yourself

```bash
# Print a reproducible queue with instant deterministic bands
uv run python -c "from triageflow.data import synthetic; from triageflow.core.scoring import score_encounter; \
[print(e.encounter_id, score_encounter(e).proposed_priority.value, '|', e.presenting_complaint) \
 for e in synthetic.stream(n=8, seed=42)]"

# One patient from a named scenario
uv run python -c "from triageflow.data.synthetic import make_scenario; print(make_scenario('asthma', 1).model_dump_json(indent=2))"

# Dump one encounter as FHIR R4 JSON
uv run python -c "import json; from triageflow.data import synthetic, fhir; \
print(json.dumps(fhir.to_bundle(synthetic.make_encounter('severe', seed=1)), indent=2, default=str))"
```
