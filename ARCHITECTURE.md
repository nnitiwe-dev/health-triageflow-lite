# Architecture

## Agent shape — a workflow, not a swarm

TriageFlow is a **LangGraph state machine with tool-calling nodes**. The triage workflow is
*predictable* (validate → extract → retrieve → score → deterioration → self-check →
approve), so a fixed graph with agent nodes is the right tool — not autonomous multi-agent
delegation. (This follows directly from the "deterministic code where you can, one graph
where you must" principle; we add no reasoning boundary we can't justify.)

```
            ┌───────────── TriageAgent (LangGraph) ─────────────┐
 Encounter ─▶ validate ─▶ extract ─▶ retrieve ─▶ score ─▶ deterioration ─▶ self_check ─▶ ⏸ approval ─▶ finalize ─▶ Confirmed
            │   (own)      (LLM+own)   (borrowed)   (own)       (own)         (LLM/det)     (human gate)   (own)
            └── validation error ────────────────────────────────────────────────────────────────▶ END (needs correction)
```

| Node | Does | Where |
|------|------|-------|
| `validate` | missing / implausible / stale / contradictory / unit checks | own MCP tool |
| `extract` | free text → **candidate** SATS signs (keyword map + LLM), nurse confirms | own MCP tool + LLM |
| `retrieve` | fetch cited SATS protocol text | **borrowed** filesystem MCP |
| `score` | **deterministic** TEWS + discriminators → sourced priority | own MCP tool |
| `deterioration` | compare re-check to baseline → second-look flag | own MCP tool |
| `self_check` | verify every point is cited and grounded in the retrieved protocol | agent (+LLM narration) |
| `approval` | `interrupt()` — pause for a named clinician | human gate |
| `finalize` | log the approval/override, return the confirmed priority | own MCP tool |

### Why the LLM never scores
The priority is a pure function of vitals + published thresholds, implemented in
`core/scoring.py`. Given the same input it returns the same output every time (proven by
`tests/` and the determinism check in `evals/run.py`). The open-weights model only
(a) proposes candidate signs from free text and (b) writes the one-line handover. Every LLM
node has a deterministic fallback, so a slow or absent model degrades gracefully rather than
breaking the run. This is what makes a priority *defensible*: zero run-to-run variance on the
number, and a citation for every point.

## MCP servers — built vs borrowed

**Built — `triageflow` (`src/triageflow/mcp/server.py`).** The agent's hands. Five tools;
four *act* (`score_triage`, `check_deterioration`, `request_clinician_approval`,
`suggest_discriminators`) and one gates data quality (`validate_triage_input`). Each call is
written to an append-only audit log (JSONL + SQLite) with inputs, outputs, timestamps and
errors. The tool boundaries are reusable by any MCP client, not just our UI.

**Borrowed — `@modelcontextprotocol/server-filesystem`** (official reference, run via
`npx`), read-only over `protocols/`. It backs `retrieve` (evidence RAG) so the self-check can
ground citations in the actual SATS text. *Why borrow over build:* sandboxed,
path-traversal-safe file access is a solved, security-sensitive problem; reusing the audited
reference is safer and faster than re-implementing it. *Why this over alternatives we
evaluated:* the official SQLite MCP is archived/unmaintained, and third-party FHIR MCPs have
unverified maintenance plus Docker friction that would threaten the one-command rule.

*(Interoperability extension, documented not required: a FHIR MCP + HAPI server is the path
to reading/writing encounters against a real EMR — our data is already FHIR R4-shaped.)*

## Control, cost & residency

- **Human-in-the-loop** is a real graph `interrupt()`, not a prompt instruction: the graph
  cannot reach `finalize` without a resume carrying a named clinician's decision.
- **Cost per run** is bounded and known: the clinical path is deterministic (0 tokens); the
  LLM adds at most two short, capped (`num_predict=120`) generations, and the model is kept
  warm (`keep_alive`). `TRIAGEFLOW_NO_LLM=1` makes it zero-token end to end.
- **Data residency:** inference is local; synthetic data only; the data region is recorded in
  config and shown in the UI. Patient data never leaves the machine.
- **Pluggable runtime:** the LLM is reached through an **OpenAI-compatible** client, so the
  backend (`TRIAGEFLOW_LLM_BACKEND`) is swappable with no code change — **Ollama** (default) or
  **llama.cpp** `llama-server`. Each backend owns its server lifecycle + model download and
  exposes a catalog so a nurse can switch to a lighter model on a low-powered box.

## Layout

```
src/triageflow/
  core/        sats_rules, scoring, models, validation, audit, config   ← deterministic heart
  data/        fhir (LOINC), synthetic generator                        ← EMR-shaped synthetic
  mcp/         server.py                                                 ← our MCP server
  agent/       graph.py (LangGraph), mcp_client.py, llm_backend + ollama/llamacpp backends  ← orchestration
  web/         app.py (FastAPI) + templates/ + static/                   ← nurse workspace
protocols/     sats_adult.md                                            ← cited evidence (borrowed MCP serves this)
tests/         pins the TEWS table + final-priority logic
evals/         cases.py + run.py (pass/fail + determinism)
```
