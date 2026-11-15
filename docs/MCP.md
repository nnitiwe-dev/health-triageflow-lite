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

All six are deterministic wrappers over the clinical core; **every call is written to the
append-only audit log**. The agent's graph decides *when* to call them; the LLM never does
their work.

### `validate_triage_input(vitals, encounter_id, age_years?) → report`
Data-quality gate. Returns `{ok, issues[], missing_required[]}`. Two tiers, on purpose:
- **error — impossible** (pulse 1300, SpO₂ 140, temperature 101 "°C", a value the schema
  rejects, an under-12 patient): scoring is blocked until it is corrected;
- **warning — extreme but possible** (breathing rate 3, systolic 35, temperature 27.5), plus
  stale measurements and simple contradictions: **scored as entered** and flagged, because
  these are the sickest patients, not typos.

Messages are written for the nurse ("Pulse rate 1300 bpm is not possible — check the entry").

### `score_triage(encounter) → result` · *action*
The core action. Deterministic SATS Adult scoring. Returns the proposed priority, TEWS total
+ band, a **sourced evidence list**, the chart's additional-investigation **`actions[]`**
(check SpO₂, glucose, ketones, ECG, pregnancy test), `missing[]`, `warnings[]`,
`ignored_codes[]`, `out_of_scope` and `complete` (false ⇒ no band is shown). Signs that
follow from a structured reading are derived here (not alert → reduced level of
consciousness; glucose ≤ 3.0 → hypoglycaemia). Anything that is not from the SATS adult chart
carries `kind: "safety_rule"` and says so in its `source`. **This is a proposal, not a
decision** — it is not confirmed until `request_clinician_approval` records a clinician.

### `check_deterioration(encounter_id, baseline_vitals, recheck_vitals) → flag` · *action*
Compares a re-check to its baseline. Returns `{escalated, baseline_tews, recheck_tews, delta,
reasons[], missing[], incomplete}`. It raises a **second-look flag** only — it never moves the
patient — when the like-for-like TEWS rose by 2 or more, the priority the vitals imply went
up, or a new red-flag finding appeared (low glucose, reduced alertness, severe pain, low
SpO₂). **It does not fail silently:** if the re-check is missing a vital, the vitals that were
measured are still compared and the gap is reported. The watch rule is a TriageFlow
convention and is cited as such, not as SATS.

### `request_clinician_approval(encounter_id, proposed_priority, clinician_name, decision, override_priority?, reason?) → record` · *action, the gate*
The human gate on the irreversible action. Enforced inside the tool, for every client:
a non-empty `clinician_name`; `approve` needs a proposed priority (an incomplete assessment
cannot be approved); `override` needs a valid `override_priority` **and** a written `reason`.
Writes an `approval` event to the audit log and returns the **confirmed** priority that may
now reach the queue.

### `suggest_discriminators(text) → {candidates[]}`
Deterministic keyword/synonym map (`core/lexicon.py`, word-boundary matched, negation-aware)
from free text to **candidate** SATS discriminator codes, each with its implied level and
source. A phrase can require two parts to be present ("pregnant" + "bleeding"). Candidates
are for the nurse to **accept or dismiss** — they never auto-apply to a score. On held-out
lay phrasing the keyword map alone has low recall (see EVALS.md); the model adds most of the
recall, and the UI always offers the full sign list.

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
  (`protocols/sats_adult.md`) for the `retrieve` step. The text is cached after the first
  read, and we fall back to a direct UTF-8 read if the server is unavailable, so evidence
  retrieval never hard-fails (the trace shows which path was used).

*(The server also exposes `list_directory`, `search_files`, write tools, etc. We only read,
and only within the `protocols/` directory passed as its allowed root.)*

---

## Call logic (how the agent uses them)

The agent is a LangGraph state machine (see ARCHITECTURE.md). The MCP calls happen in a fixed,
auditable order — this is a *workflow*, not a free-running loop:

```
validate_triage_input        (triageflow)      → data-quality gate; impossible values end the run
read_text_file               (filesystem)      → protocol text for the self-check (cached)
score_triage                 (triageflow)      → deterministic priority + evidence + actions
check_deterioration          (triageflow)      → (only on a re-check) second-look flag
   — self-check, then the proposal is shown to the nurse —
suggest_discriminators       (triageflow)      → candidate signs (+ the model's, with a timeout)
   ⏸  interrupt()  — human approval gate
request_clinician_approval   (triageflow)      → logs the named decision; returns confirmed priority
```

**Gating rule:** nothing downstream of the `interrupt()` runs until a clinician's decision is
supplied, and the decision is checked first (named approver, a priority to approve, every
suggested sign reviewed, the sign list checked when the model did not read the complaint,
self-check passed; or a reasoned override). The gate is structural, not a prompt instruction.

**Determinism & cost:** `score_triage` / `check_deterioration` are pure functions (0 tokens,
identical output for identical input). The only LLM usage is candidate suggestion (capped at
`max_tokens=120`, with a timeout), skippable with `TRIAGEFLOW_NO_LLM=1`, plus an opt-in
handover note outside the gate path (`TRIAGEFLOW_HANDOVER_NOTE=1`, off by default).

**Start-up:** only our own server is awaited. The borrowed server is attached in the
background with a timeout, so a slow or absent `npx` download never delays the desk.

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
