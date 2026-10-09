# Evaluation

Run it yourself:

```bash
uv run python evals/run.py      # scenario pass/fail + determinism
uv run python -m pytest -q      # 57 unit tests pinning the SATS/TEWS table
```

The set probes behaviour honestly — under-triage, missing data, bad units, overrides, and
deterioration — rather than being curated to pass. Cases and expectations are in
`evals/cases.py`; `E8` is a **known failure we did not fix** (see below).

## Test tasks & results

| # | Scenario | Expected | Result |
|---|----------|----------|--------|
| E1 | Complete healthy adult | GREEN, complete, ≥7 citations | ✅ pass |
| E2 | Deranged vitals (septic-looking), RR 32 / HR 132 / SBP 88 / T 39.6 / confused | RED by TEWS | ✅ pass |
| E3 | Chest pain, **normal** vitals | ORANGE via discriminator (not TEWS) | ✅ pass |
| E4 | Emergency sign (cardiac arrest) with **incomplete** vitals | RED, still "complete" (sign forces it) | ✅ pass |
| E5 | Missing required vitals, no overriding sign | **Incomplete** — no band shown | ✅ pass |
| E6 | Otherwise-green patient, SpO₂ 88% | ORANGE via additional-investigation flag | ✅ pass |
| E7 | ORANGE patient re-checked and worsened | YELLOW baseline; **deterioration flag** raised | ✅ pass |
| E8 | Free-text-only emergency, keyword miss | ORANGE (clinically) | ❌ **xfail** (system gives GREEN) |
| E9 | Implausible HR = 400 | validation ERROR | ✅ pass |
| E10 | Temperature entered as 101.2 (Fahrenheit) | validation ERROR (unit catch) | ✅ pass |

**10 cases · 9 pass · 1 documented known-failure · 0 regressions.**

What's checked per case: final priority, completeness (we never show a band on insufficient
data), citation count (every priority is sourced), the missing-vitals list, and — for E7 —
that `check_deterioration` escalates.

## Reliability: run-to-run variance

The harness scores each case **25×** and asserts the serialized result is byte-identical.
All scored cases are **deterministic** (`DETERMINISTIC: yes`). This is by design — the
clinical priority is pure code, not a model sample, so there is no variance to hide. The LLM
(candidate extraction + narration) is kept off the scoring path; its variance affects
*suggestions a nurse confirms*, never the number.

## The failure we did NOT fix (E8)

**Scenario.** Complaint is free text only — *"crushing retrosternal discomfort radiating to
jaw"* — with normal vitals and **no** structured sign selected. Clinically this is chest
pain → **ORANGE**. The deterministic engine scores TEWS 0 and, with no `chest_pain`
discriminator confirmed, returns **GREEN**. That is **under-triage** — the most dangerous
error class.

**Why it happens.** The score only counts *confirmed* discriminators (a deliberate safety
choice: we never auto-apply an unconfirmed sign). The `suggest_discriminators` keyword map
has no token matching "crushing retrosternal discomfort," so nothing is surfaced for the
nurse to confirm, and the LLM extractor — when enabled — is not guaranteed to map this
phrasing to `chest_pain` either.

**Why we left it.** Fixing it by auto-applying LLM-inferred signs would cross the line the
challenge draws (an unsourced/over-confident flag driving priority). The safe fix is better
*surfacing*, not silent escalation.

**Partial mitigation shipped.** The symptom lexicon (`core/lexicon.py`) was expanded with lay
+ clinical synonyms and word-boundary matching, so this exact phrasing now **surfaces
`chest_pain` as a candidate** for the nurse to confirm (see `tests/test_lexicon.py`). The
deterministic *score* still returns GREEN until the sign is confirmed — by design, we never
auto-apply an unconfirmed sign — so E8 remains an honest `xfail`. The residual gap is the
human step, not silent escalation.

**What we would try next.**
1. ✅ (done) Expand the keyword/synonym lexicon so common descriptions of red-flag complaints
   are at least *suggested*. Next: measure its recall as a first-class metric (below).
2. Add a constrained-decoding LLM pass (JSON-schema enum over the discriminator catalog) with
   a confidence threshold, surfaced as a **highlighted** candidate the nurse must accept.
3. Add an eval sub-suite of paraphrased red-flag complaints and track **sensitivity** (recall
   of the correct discriminator) as a first-class metric — under-triage recall matters more
   than overall accuracy here.
