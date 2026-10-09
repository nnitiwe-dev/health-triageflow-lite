# Contributing to TriageFlow Lite

Thanks for your interest. This is a clinical-adjacent tool, so a few rules below are stricter
than a typical project — they keep the "sourced, nurse-confirmed, never a diagnosis" promise
intact.

## Ground rules (non-negotiable)

1. **Never add real patient data**, even de-identified. Synthetic only.
2. **The LLM must never compute a priority.** Clinical scoring stays deterministic
   (`core/scoring.py`). The model may only extract/suggest and narrate.
3. **Every priority must stay sourced.** Any new rule carries a citation; any new flag cites
   the vital/sign behind it.
4. **Changes to clinical thresholds** (`core/sats_rules.py`, `protocols/sats_adult.md`) must
   cite the source edition and should be reviewed by someone clinically qualified. Bump
   `RULESET_VERSION` when thresholds change.
5. **Don't add a dependency on restricted content** (e.g. SNOMED CT) without a licence review.

## Dev setup

```bash
uv run triageflow doctor         # checks Ollama + model + both MCP servers
uv run --extra dev python -m pytest   # unit tests
uv run python evals/run.py            # scenario evals + determinism
uv run --extra dev ruff check src tests   # lint
```

Python ≥ 3.11. The project uses `uv`; `uv run` auto-syncs the environment.

## Making a change

- Keep PRs small and scoped. Match the surrounding code's style and comment density.
- Add/adjust tests: new TEWS logic → `tests/test_scoring.py`; new lexicon phrases →
  `tests/test_lexicon.py`; new behaviour → an `evals/cases.py` case.
- Run `pytest`, `evals/run.py`, and `ruff` before opening the PR. CI runs all three.
- Update docs if you change behaviour (README / ARCHITECTURE / MCP / DATA / EVALS).

### Adding a symptom phrase
Edit `SYMPTOM_LEXICON` in `src/triageflow/core/lexicon.py`. The code key must be a valid
discriminator in `sats_rules.py`. Prefer **recall** for red-flags (surface a candidate), and
add a case to `tests/test_lexicon.py`. Candidates are always nurse-confirmed, never auto-applied.

### Adding an MCP tool
Add it to `src/triageflow/mcp/server.py`, log the call to the audit log, keep it deterministic
where the decision can be expressed as a rule, document it in `docs/MCP.md`, and wire it into
the graph only if the workflow needs it.

## Reporting bugs / ideas
Open an issue with repro steps (a seed + encounter is ideal, since generation is reproducible).
For anything security- or privacy-sensitive, see [SECURITY.md](SECURITY.md) instead.

## Licence
By contributing you agree your contributions are licensed under the project's **Apache-2.0**
licence.
