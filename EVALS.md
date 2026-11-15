# Evaluation

Run it yourself:

```bash
uv run python evals/run.py          # scoring, approval gate, synthetic set, keyword recall
uv run python evals/run.py --llm    # the same, plus sign-finding with the local model
uv run python evals/runtime_smoke.py   # llama.cpp for real: provision, start, generate, stop
uv run python -m pytest -q          # 340 tests
```

The set probes behaviour honestly — under-triage, missing and impossible data, the approval
gate, and how well signs are found in free text — rather than being curated to pass. Cases
and expectations are in `evals/cases.py`. **X1 is a known failure we did not fix** (below).

## 1. Scoring — is the SATS priority right? (18 / 18)

| # | Scenario | Expected | Result |
|---|----------|----------|--------|
| E1 | Complete healthy adult | GREEN, complete, ≥ 7 citations | ✅ |
| E2 | Septic-looking: RR 32 / HR 132 / SBP 88 / T 39.6 / confused / immobile | RED by TEWS (13) | ✅ |
| E3 | Chest pain, **normal** vitals | ORANGE via the sign, plus an ECG action | ✅ |
| E4 | Emergency sign with **no vitals at all** | RED — a TEWS is not required | ✅ |
| E5 | Missing vitals, no overriding sign | **Incomplete** — no band shown | ✅ |
| E6 | SpO₂ 88%, otherwise normal | ORANGE, labelled a **local rule, not SATS** | ✅ |
| E7 | Urgent patient re-checked and worse | deterioration flag | ✅ |
| E8 | Re-check clearly worse, temperature **not retaken** | flag still raised (not silent) | ✅ |
| E9 | Impossible HR = 400 | validation ERROR | ✅ |
| E10 | Temperature entered as 101.2 (Fahrenheit) | validation ERROR (unit catch) | ✅ |
| E11 | **Confused**, vitals otherwise normal | ORANGE (SATS: not fully alert) + glucose action | ✅ |
| E12 | Responds only to voice, walking | ORANGE (TEWS alone would say GREEN) | ✅ |
| E13 | Glucose exactly 3.0 mmol/L | RED (chart: "3 or less") | ✅ |
| E14 | Glucose 28, no diabetic sign confirmed | not escalated; ketone check prompted | ✅ |
| E15 | Breathing rate 3 /min | **scored** with a warning — not rejected as a typo | ✅ |
| E16 | 4-year-old | out of scope — no adult band | ✅ |
| E17 | RR 14.5, HR 100.5 (between printed bands) | scored conservatively, not "missing" | ✅ |
| E18 | Pain 8/10 | ORANGE; cut-off cited as a convention | ✅ |

E11–E17 are failures an audit of the first version found (a confused patient scored GREEN; a
glucose of 3.0 scored nothing; RR 3 was rejected as a typo; a child got an adult score; a
decimal reading was reported as missing). They are now pinned.

## 2. Flow — can an unsafe result reach the queue? (8 / 8)

These run the whole agent (validate → score → self-check → extract → gate) and then try to
approve.

| # | Attempt | Expected | Result |
|---|---------|----------|--------|
| F1 | "Chest pain" typed, normal vitals, suggestion **not reviewed** → approve | refused | ✅ |
| F2 | Same patient, nurse marks chest pain present → approve | ORANGE confirmed | ✅ |
| F3 | Incomplete assessment → approve | refused | ✅ |
| F4 | Override RED → GREEN with no reason | refused | ✅ |
| F5 | Mistyped vital (HR 1300) → approve | nothing to approve; no stale result | ✅ |
| F6 | Emergency sign ticked **and** a mistyped vital | RED confirmed — not held up | ✅ |
| F7 | Model did not read the complaint, sign list not checked → approve | refused | ✅ |
| F8 | No approver name | refused | ✅ |

## 3. Synthetic set (1,300 / 1,300)

All 26 scenarios × 50 seeds score to their intended SATS colour. (The test-suite additionally
checks that no scenario raises a validation issue and that the keyword matcher surfaces every
sign each note describes — see `tests/test_synthetic.py`.)

## 4. Sign-finding in free text — measured, not assumed

`PARAPHRASES` is a **held-out** set: 36 lay descriptions of red-flag complaints ("says his
chest hurts and left arm feels heavy", "knocked off his motorbike by a lorry", "five months
pregnant with bleeding from below") written *after* the keyword lexicon and never used to
tune it. `BENIGN` is 10 routine notes full of pertinent negatives.

| Extractor | Recall on 36 held-out phrasings | Routine notes wrongly flagged | Time per complaint |
|-----------|--------------------------------:|------------------------------:|--------------------|
| Keyword map only (model off) | **5 / 36 = 14%** | 0 / 10 | instant |
| Qwen2.5 7B (Q4) on llama.cpp, CPU | **29 / 36 = 81%** | 0 / 10 | median 3.6 s, slowest 24.5 s |
| Model + keywords (what the app offers) | **31 / 36 = 86%** | 0 / 10 | — |

Measured on an Intel Mac (CPU only) with llama.cpp build b12359; the combined row is the
union of the two runs. The model was also running alongside another process for part of the
run, so the slowest time is pessimistic.

What the model still missed: "pressure in the middle of the chest", "asthmatic, chest tight
and wheezy", "suddenly cannot lift the right arm", "bringing up blood when she coughs"
(offered *vomiting* blood instead), "shoulder popped out playing rugby". It also sometimes
adds a wrong extra sign (e.g. *severe pain* for a heavy arm) — which is why every suggestion
must be accepted or dismissed by the nurse and none is ever applied automatically.

**Reading these numbers.** The keyword map is a floor, not the feature: it is exact on
clinical wording ("haematemesis", "post-ictal", "RTA") and on the synthetic notes, and weak on
everyday speech. The model supplies most of the recall. Neither is trusted to decide: the
full sign list is always available, and the gate asks the nurse to confirm she has checked it
whenever the model has not read the complaint.

### A second model finding: the handover note

The first design let the model write a one-line handover. While capturing the README
screenshot the 7B model wrote *"Finger-prick glucose test done, ECG pending"* for a patient
whose result only *prompted* those checks. It had passed the note's checks (right colour, no
diagnosis). The note is therefore **off by default**; the handover is the summary generated
by code. If switched on, the model is no longer shown the actions and a note that claims
anything was done, given or ordered is discarded (`tests/test_agent_flow.py`).

## 5. Reliability: run-to-run variance

The harness scores each scoring case **25×** and asserts the serialized result is
byte-identical. All are **deterministic**. This is by design — the clinical priority is pure
code, not a model sample, so there is no variance to hide. The model is kept off the scoring
path; its variance affects *suggestions a nurse reviews*, never the number.

## 6. The failure we did NOT fix (X1)

**Scenario.** The language model is off (not downloaded, still loading, or disabled). The
nurse types *"feels like an elephant sitting on his chest, sweating a lot"*, vitals normal.
Clinically this is chest pain → **ORANGE**. The keyword map finds nothing, so nothing is
suggested and the engine proposes **GREEN**. That is **under-triage** — the most dangerous
error class — and with the model off it will happen for most lay phrasings (14% recall).

**Why we left it.** The only ways to make the *system* catch it are to apply signs the model
infers without a human (an unsourced flag driving a priority — the line this project does
not cross) or to keep growing a keyword list, which the held-out measurement shows does not
generalise. With the model on, this phrasing is caught; with it off, a person has to.

**What limits the harm today** (each is tested):
1. The screen says so, in words, next to the suggestions: *"Language model is off — only
   keyword matching was used. Check the full sign list…"*.
2. The **full SATS sign list** is one click away and a tick re-scores immediately.
3. The gate **refuses** a plain approval until the nurse ticks *"I have checked the full sign
   list"* (flow case F7). It is an attestation, not a guarantee: a nurse who ticks without
   looking will still under-triage this patient.

**What we would try next.**
1. Constrained decoding (a JSON-schema enum over the sign catalogue) to remove wrong extras
   and stabilise the smaller 3B / 0.5B models, then publish their recall next to the 7B's.
2. Local-language and regional phrasings collected with triage nurses, added to
   `PARAPHRASES` first (to measure) and only then to the lexicon.
3. Track held-out recall per release, split by sign, with the emergency and very-urgent
   signs reported separately — under-triage recall matters more than overall accuracy here.
