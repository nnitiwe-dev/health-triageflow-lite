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
- **Residency.** Inference runs locally (Ollama); the configured data region is recorded in
  config and shown in the UI header. Nothing is sent to an external service.

## 2. What is generated

Adult (>12y / >150cm) triage encounters, each with:

| Part | Fields |
|------|--------|
| Demographics | age, sex (synthetic) |
| Presenting complaint | free-text string + 0–n SATS discriminator codes |
| Vitals | respiratory rate, heart rate, systolic BP, temperature, AVPU, mobility, trauma |
| Additional investigations | SpO₂, finger-prick glucose (optional) |
| Timing | arrival time, measurement time |

Produced as:
- **Native FHIR R4** resources (`Patient`, `Encounter`, `Condition`, `Observation`) so an
  encounter looks like what a real EMR/FHIR API returns (`src/triageflow/data/fhir.py`).
- A reproducible **waiting queue** and a **deterioration series** (baseline → worsened
  re-check) for the deterioration-watch demo (`src/triageflow/data/synthetic.py`).

## 3. How it is generated (method & explainability)

Each encounter is drawn from one of four **acuity profiles** — `stable`, `moderate`,
`severe`, `critical`. For every profile:

1. Continuous vitals are sampled from a **Gaussian** around a clinically plausible mean/spread
   for that acuity, then **clipped** to hard physiological bounds (e.g. HR 35–180). Example:
   a `critical` patient draws RR ~ N(34, 6), HR ~ N(135, 18), SpO₂ ~ N(86, 4).
2. Categorical vitals (AVPU, mobility) are sampled from **profile-specific weights** (e.g. a
   `critical` patient is weighted toward "reacts to pain / unresponsive" and "immobile").
3. The presenting complaint + discriminator codes are chosen from a curated list **matched to
   that acuity**, so the free text is coherent with the vitals.

This means the data is **not** label-leaking: vitals and complaint are generated from the
profile, and the SATS priority is then computed independently by the deterministic engine —
the data does not carry its own "answer".

**Reproducibility.** Everything is **seeded** (`stream(seed=…)`, `make_encounter(seed=…)`).
The same seed yields the same queue on any machine — essential for stable evals and for a
reviewer to reproduce a run exactly.

## 4. Why you can trust it (and audit it)

- **Transparent & inspectable.** The generator is ~120 lines of readable Python with the
  distributions written out explicitly; there is no opaque model producing the data.
- **Deterministic pipeline.** Data in → the deterministic SATS engine → a priority with a
  citation for every point. Given the same encounter, the same priority, every time.
- **FHIR-validatable.** Bundles can be round-tripped (`vitals_from_observations`) and fed to a
  real FHIR server (interoperability extension in ARCHITECTURE.md) for structural validation.
- **Audit trail.** Every assessment (tool calls, inputs/outputs, timestamps, and the named
  clinician approval) is written append-only to `~/.triageflow/audit.{jsonl,db}`.

## 5. Limitations & non-use (important)

- **Face-validity, not clinical realism.** Distributions are chosen to be *plausible* and to
  exercise every SATS pathway — they are **not** calibrated to any specific population's
  epidemiology, and correlations between vitals are simplified (each vital is sampled largely
  independently within a profile).
- **Not a validation set.** Passing evals on this data demonstrates the *engine behaves as
  specified*; it does **not** establish clinical safety or accuracy on real patients.
- **Adult only.** Paediatric physiology and the paediatric SATS pathway are out of scope.
- **Do not substitute for consented real data** when moving toward a pilot. A facility pilot
  must use the locally-adopted protocol and data governed by that facility's consent/ethics
  process — never real patient data without consent, even de-identified.

## 6. Reproduce / inspect it yourself

```bash
# Print a reproducible queue with instant deterministic bands
uv run python -c "from triageflow.data import synthetic; from triageflow.core.scoring import score_encounter; \
[print(e.encounter_id, score_encounter(e).proposed_priority.value, '|', e.presenting_complaint) \
 for e in synthetic.stream(n=8, seed=42)]"

# Dump one encounter as FHIR R4 JSON
uv run python -c "import json; from triageflow.data import synthetic, fhir; \
print(json.dumps(fhir.to_bundle(synthetic.make_encounter('severe', seed=1)), indent=2, default=str))"
```
