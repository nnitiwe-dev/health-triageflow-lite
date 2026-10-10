# MCP reference — servers, tools, and call logic

TriageFlow speaks to **two** MCP servers through the standard `langchain-mcp-adapters`
client (`src/triageflow/agent/mcp_client.py`), so the same code would work against any other
MCP server a facility already runs.

| Server | Origin | Transport | Role |
|--------|--------|-----------|------|
| `triageflow` | **built by us** (`src/triageflow/mcp/server.py`) | stdio | the agent's *hands* — validate, score, flag, gate |
| `filesystem` | **borrowed** — official `@modelcontextprotocol/server-filesystem` via `npx` | stdio | read-only evidence retrieval over `protocols/` |

**Why borrow the filesystem server (one line):** sandboxed, path-traversal-safe file access is
a solved, security-sensitive problem — reusing the audited reference implementation is safer
and faster than writing our own, and it keeps evidence retrieval on a standard interface.

Run `uv run triageflow doctor` to see both servers connect and list their tools.

---

## Built server: `triageflow` tools

All five are deterministic wrappers over the clinical core; **every call is written to the
append-only audit log**. The LLM decides *when* to call them; it never does their work.

### `validate_triage_input(vitals, encounter_id) → report`
Data-quality gate. Returns `{ok, issues[], missing_required[]}`. Flags missing required
vitals, implausible values, wrong units (e.g. Fahrenheit temperature), stale measurements,
and simple contradictions. Severity is `error` (blocks a defensible score) or `warning`.

### `score_triage(encounter) → result` · *action*
The core action. Deterministic SATS Adult scoring. Returns the proposed priority, TEWS total
+ band, a **sourced evidence list** (each point cites the SATS chart), `missing[]`,
`warnings[]`, and `complete` (false ⇒ no band is shown). **This is a proposal, not a
decision** — it is not confirmed until `request_clinician_approval` records a clinician.

### `check_deterioration(encounter_id, baseline_vitals, recheck_vitals) → flag` · *action*
Compares a re-check to baseline. Returns `{escalated, baseline_tews, recheck_tews, delta,
reasons[]}` where each reason is sourced. Raises a **second-look flag** only — it never moves
the patient. Escalation triggers: TEWS rose ≥2, band escalated, or SpO₂ dropped below 92%.

### `request_clinician_approval(encounter_id, proposed_priority, clinician_name, decision, override_priority?, reason?) → record` · *action, the gate*
The human gate on the irreversible action. `decision ∈ {approve, override}`; a non-empty
`clinician_name` is required (the gate needs a named approver); `override` requires an
`override_priority`. Writes an `approval` event to the audit log and returns the **confirmed**
priority that may now reach the queue.

### `suggest_discriminators(text) → {candidates[]}`
Deterministic keyword/synonym map (`core/lexicon.py`, word-boundary matched, negation-aware)
from free text to **candidate** SATS discriminator codes, each with its implied level and
source. Candidates are for the nurse to **confirm** — they never auto-apply to a score.

### `fetch_fhir_vitals(patient_id) → {vitals, source}`  *(optional, EMR ingest)*
Imports a patient's vital-signs Observations from the configured FHIR R4 server and maps them
to our schema, so the nurse doesn't re-type what the EMR already holds. Disabled unless
`TRIAGEFLOW_FHIR_BASE_URL` is set; the result is still nurse-confirmed. See the FHIR note below.

## FHIR / EMR interoperability (and why it is a *tool*, not a borrowed MCP)

We looked hard for a borrowed **FHIR MCP**: the maintained ones (`the-momentum`, `fhirhydrant`)
assume OAuth2/SMART + Docker and won't cleanly hit an unauthenticated sandbox, and the one
lightweight optional-auth option's repo is gone. Depending on any of them would risk the
"clone-and-run" requirement. So EMR ingest is a reliable **first-party tool**
(`fetch_fhir_vitals` + `agent/fhir_emr.py`) over plain FHIR REST, **off by default**, pointed at
a configurable base URL. Our synthetic data is already FHIR R4 + LOINC, so a real write→read
round trip works against any R4 server. **For a pilot:** point `TRIAGEFLOW_FHIR_BASE_URL` at the
facility's in-country FHIR server (optionally fronted by a FHIR MCP) — same code path.

## Borrowed server: `filesystem` (tools used)

- `read_text_file(path)` / `read_file(path)` — fetch the cited SATS protocol text
  (`protocols/sats_adult.md`) for the `retrieve` step. We fall back to a direct read if the
  server is unavailable, so evidence retrieval never hard-fails.

*(The server also exposes `list_directory`, `search_files`, write tools, etc. We only read,
and only within the `protocols/` directory passed as its allowed root.)*

---

## Call logic (how the agent uses them)

The agent is a LangGraph state machine (see ARCHITECTURE.md). The MCP calls happen in a fixed,
auditable order — this is a *workflow*, not a free-running loop:

```
validate_triage_input        (triageflow)      → data-quality gate; errors short-circuit to END
suggest_discriminators       (triageflow)      → candidates surfaced for nurse confirmation
read_text_file               (filesystem)      → protocol evidence for grounding
score_triage                 (triageflow)      → deterministic priority + sourced evidence
check_deterioration          (triageflow)      → (only if a re-check exists) second-look flag
   ⏸  interrupt()  — human approval gate
request_clinician_approval   (triageflow)      → logs the named decision; returns confirmed priority
```

**Gating rule:** nothing downstream of the `interrupt()` runs until a clinician's decision is
supplied. The graph literally cannot reach `request_clinician_approval` without a named
approver — the gate is structural, not a prompt instruction.

**Determinism & cost:** `score_triage` / `check_deterioration` are pure functions (0 tokens,
identical output for identical input). The only LLM usage is candidate refinement and the
one-line handover, both capped (`num_predict=120`) and skippable (`TRIAGEFLOW_NO_LLM=1`).

## Reuse by other clients

Because these are real MCP tools, any MCP client (another agent, Claude Desktop, an internal
app) can consume them. Minimal client config:

```jsonc
{
  "mcpServers": {
    "triageflow": { "command": "python", "args": ["-m", "triageflow.mcp.server"] },
    "filesystem": { "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "./protocols"] }
  }
}
```
