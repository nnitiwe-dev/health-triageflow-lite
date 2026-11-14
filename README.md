# TriageFlow Lite

**An agentic assistant for the busy OPD triage desk.** It turns a patient's vitals and
presenting complaint into a **sourced SATS priority level** in seconds — so the nurse can
see who needs to be looked at next, backed by the numbers already in front of her.

> **The one rule.** TriageFlow *scores and sequences* — it **never diagnoses, never
> prescribes, and never moves a patient**. A named nurse or clinician confirms every
> priority before it reaches a queue position. Every priority cites the vital or sign
> behind it; an unsourced flag is worse than no flag.

Theme: **Symptom-based priority scoring** · Protocol: **South African Triage Scale (SATS) Adult / TEWS**

![TriageFlow Lite — the nurse workspace: the waiting queue sorted by urgency with a confirmed patient in it, intake with the emergency-sign ticks, the suggested signs marked present, the agent's live tool calls across both MCP servers, and an ORANGE priority with its reasons and "do now" actions.](docs/img/screenshot.png)

<p align="center"><sub>The nurse workspace — intake (emergency signs first), signs to confirm with the agent's live tool-call trace, and the proposed priority with its reasons and "do now" actions, awaiting confirmation.</sub></p>

### What this README covers

| Area | Where |
|------|-------|
| Problem statement | [The problem](#the-problem) · [Grounded in a real triage desk](#grounded-in-a-real-triage-desk) |
| Solution overview · Target users | [What it does](#what-it-does-the-workflow-it-serves) |
| Setup / installation · Usage | [Run it](#run-it-one-command) |
| Architecture · Agent architecture | [Architecture](#architecture) · [ARCHITECTURE.md](ARCHITECTURE.md) |
| MCP implementation · tools / servers | [MCP servers & tools](#mcp-servers--tools) · [docs/MCP.md](docs/MCP.md) |
| Technology stack | [Technology stack](#technology-stack) |
| Human-in-the-loop workflow | [Human-in-the-loop](#human-in-the-loop) |
| Limitations · Future improvements | [Limitations](#limitations) · [Future improvements](#future-improvements) |

---

## The problem

In most public-hospital outpatient departments across Africa the bottleneck is **triage**, not
the doctor's door. One or two nurses with a BP cuff and a stack of case cards work through
everyone before anyone gets a priority — so a genuinely urgent patient can wait **60–90 minutes**
in the same queue as a routine follow-up, because nothing flags who to see next. **The wait
itself becomes where the risk is.** A practising Nigerian nurse we spoke with confirmed many
hospitals have *no formal triage framework at all*. **Who experiences it:** the triage nurse at
that single desk (overloaded, no decision support) and the patients whose time-critical
conditions stay invisible until too late.

---

## Run it (one command)

The only prerequisite is [`uv`](https://docs.astral.sh/uv/). It works the same on **Windows,
macOS and Linux**.

```bash
uv run triageflow serve
```

Open <http://127.0.0.1:8000>. The page is up in a couple of seconds and **priorities work
immediately**; everything else attaches in the background:

- **Model runtime.** If neither llama.cpp nor Ollama is on the machine, the official prebuilt
  `llama-server` for your OS/CPU (about 25 MB) is downloaded, unpacked and test-run
  automatically. No Homebrew, no compiler, no admin rights.
- **Model.** Pick one from the **model:** menu (or `triageflow pull-model`). A model Ollama
  already downloaded is **reused in place** — no second multi-GB download.
- **Borrowed MCP.** If [Node.js](https://nodejs.org) is installed, `npx` fetches the
  filesystem MCP; if not (or you are offline), the protocol text is read locally instead.

Other commands:

```bash
uv run triageflow doctor        # platform, model runtime, model, both MCP servers
uv run triageflow pull-model    # download a model (resumable); add an id to choose one
uv run triageflow demo          # one assessment in the terminal: tool calls, review, approval
uv run python evals/run.py      # evaluation suite (add --llm to measure the model too)
uv run python -m pytest         # tests: SATS table, approval gate, runtime provisioning
```

**No GPU needed.** The clinical priority is computed by deterministic code and is on screen
within a few seconds — **before any model call**. The open-weights model only *suggests* signs
from the free-text complaint (for the nurse to accept or dismiss). Set `TRIAGEFLOW_NO_LLM=1`
to run with no model at all.

### Models & backends (pick a lighter model, or swap the runtime)

The LLM runs behind an **OpenAI-compatible** interface, so the runtime is pluggable
(`TRIAGEFLOW_LLM_BACKEND`), and the agent code never changes.

- **llama.cpp — the primary runtime** (`llama-server`: one small binary, no daemon, no GUI).
  Self-provisioned on every OS from the official llama.cpp release that the current *stable*
  version points to; the binary is verified by running it, and a clear message is shown if it
  cannot run on the machine. GGUF models come from Hugging Face on demand — standard **7B**,
  or a **3B** / tiny **0.5B** for low-end hardware — with resume + retry, a disk-space check
  and a file-integrity check. If port 8080 is taken, a free port is used.
- **Ollama** — used automatically if it is already installed and llama.cpp is not (no
  redundant setup). Install from [ollama.com](https://ollama.com/download).
- **`auto` (default)** — llama.cpp if present → else an installed Ollama → else set up
  llama.cpp. Force one with `TRIAGEFLOW_LLM_BACKEND=llamacpp` or `=ollama`.
- **Switch models from the UI** — the **model:** menu lists the catalog with download sizes;
  on a low-powered machine pick a smaller one and it downloads + switches with live progress.
  Same from the terminal: `uv run triageflow pull-model qwen2.5-3b-instruct-q4`.

`uv run triageflow doctor` shows the platform, active backend, resolved model and catalog.
The `llama-server` log is kept at `~/.triageflow/logs/llama-server.log`.

### Optional: import vitals from a FHIR EMR
Off by default (the app runs fully local, no network — data residency). Point it at a FHIR R4
server to let the nurse **import a patient's vitals** instead of re-typing them:

```bash
TRIAGEFLOW_FHIR_BASE_URL=https://hapi.fhir.org/baseR4 uv run triageflow serve
```

An **Import from EMR** button appears; it does a real write→read round trip (seeds a synthetic
patient, reads the vitals back) and populates intake for the nurse to confirm. All data is
**synthetic** — for a pilot, point the URL at the facility's **in-country** FHIR server so
patient data never leaves the country. If the server is unreachable, the manual/synthetic path
is unaffected. See `uv run triageflow seed-fhir -n 3` to pre-seed patients, and `doctor` to check it.

---

## Grounded in a real triage desk

We spoke with a **practising Nigerian nurse** (Oct 2026). Two things stood out:

> *"Not all hospitals triage patients to begin with … there isn't [a formal framework]."* — and
> *"The most severe cases go in first; it's not first-come-first-serve."*

That validated the problem (triage is inconsistent/absent in many Nigerian OPDs) and the
design (acuity-sorted queue, not FCFS). She also gave the exact data a nurse records. We
changed the tool to match it, without touching the SATS scoring logic:

| Nurse said | What we changed |
|------------|-----------------|
| "I document both systolic **and** diastolic" | Added **diastolic BP** capture (recorded; TEWS still scores systolic, per SATS) |
| "We use **pulse rate**" | Relabelled Heart rate → **Pulse rate** |
| "**Pain score 0–10**" | Added pain 0–10, offered as the **SATS pain sign** (4–6 → moderate/YELLOW, 7–10 → severe/ORANGE). The cut-offs are our convention — SATS defines severe pain as "unbearable" — so the result says so and the nurse confirms |
| "AVPU **or GCS**", "weight" | Added **weight**; kept AVPU (the SATS standard); GCS→AVPU mapping is noted as future work |

## What it does (the workflow it serves)

**Facility:** a public hospital outpatient department. **User:** the triage nurse at the
single desk that everyone passes through before they are assigned a priority.

The screen follows the **SATS five steps** in the order the nurse works:

1. **Emergency signs first.** Four tick-boxes at the top of intake (not breathing, cardiac
   arrest, seizing now, burn to face/airway). One tick gives **RED** at once — no vitals, no
   typing, exactly as SATS requires.
2. **Complaint and vitals.** She types the complaint in her own words and enters what she
   measured, then clicks **Assess**. The agent, visibly:
   - **validates** the entry — an impossible value (pulse 1300, temperature 101 "°C") stops
     with a plain message; an extreme but possible one (breathing rate 3) is scored and
     flagged, never thrown away as a typo;
   - **retrieves** the protocol text from the borrowed filesystem MCP;
   - **scores** the SATS priority deterministically and shows it straight away;
   - **checks itself** — every point cited, every band and sign matched to the protocol text;
   - **suggests signs** found in the complaint (keyword map + the model).
3. **Signs to confirm.** Each suggestion must be marked **Present** or **Not present**; until
   then the result is shown as *provisional*. The **full SATS sign list** is always one click
   away for anything the suggestions missed. Signs that follow from a reading are applied
   directly (not fully alert → very urgent; glucose 3.0 or less → emergency).
4. **Do now.** The chart's own "additional investigations" appear as actions — measure SpO₂,
   finger-prick glucose, ketone dipstick, ECG, pregnancy test — each with why it applies.
5. **Approval gate.** Nothing is queued until she confirms or a senior clinician overrides.

A confirmed patient appears in the **waiting queue**, ordered by urgency. **Re-check vitals
later** compares a new set of vitals with the first and raises a **second-look flag** if the
patient is worse — it still cannot move them.

### How a result reads

| On screen | What it tells you |
|-----------|-------------------|
| **ORANGE — Very urgent** · *See within 10 minutes* · Majors | The priority, the time target and where to send the patient |
| **Why this priority** | Each sign and abnormal vital in plain words (*"Fast breathing — 28 /min, +2 TEWS"*); the one that **decides the priority** is marked |
| **Do now** | The SATS investigations/tasks triggered for this patient |
| **Check** | Anything to double-check: an extreme reading, missing age, stale vitals |
| **TEWS working** / **Sources** | The full points table and the citation for every rule — folded away until wanted |
| Badges | *every point cited* · *matches the protocol text* · *local safety rule (not SATS)* when one applied |

Anything that is **not** from the SATS adult chart is labelled as such. Today that is one
rule: an adult **SpO₂ below 92%** is raised to at least ORANGE as a local safety rule (the
adult chart only says to check SpO₂ and hand over for oxygen).

## Human-in-the-loop

The approval gate is **structural, not a prompt instruction**: the agent is a LangGraph graph
that `interrupt()`s before its final step and **cannot reach `finalize` without a named
clinician's decision**. The rules are enforced **on the server** — no client can skip them:

- no approval without a **named** clinician;
- no approval of an **incomplete** assessment (it has no priority);
- no approval while a **suggested sign is unreviewed**;
- if the language model did **not** read the complaint (off, slow, or loading), the nurse must
  confirm she has **checked the sign list** herself;
- no approval if the **self-check failed**;
- an **override** needs a valid priority and a **written reason** (senior clinician's discretion).

A refusal comes back as a sentence the nurse can act on — never as a blank or a false
"confirmed". Each assessment is its own run, so a result on screen can never be a leftover
from an earlier one; editing the intake after a result dims it and locks the gate until it is
re-assessed. Every tool call and the named decision are written to an append-only audit log.

---

## Documentation

| Doc | What's in it |
|-----|--------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Agent shape, nodes, MCP built-vs-borrowed, control/cost/residency |
| [docs/MCP.md](docs/MCP.md) | The two MCP servers, full tool catalog + schemas, call logic |
| [docs/DATA.md](docs/DATA.md) | **Stakeholder data card** — how synthetic data is generated, explainability & trust, limitations |
| [EVALS.md](EVALS.md) | Scoring, approval-gate and synthetic-set results, sign-finding recall, the documented unfixed failure |
| [DESCRIPTION.md](DESCRIPTION.md) | ~300-word summary (theme, facility, workflow) |
| [CONTRIBUTING.md](CONTRIBUTING.md) · [SECURITY.md](SECURITY.md) · [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) | Contributor guide, privacy/disclosure policy, conduct |

## Architecture

A **LangGraph** state machine with tool-calling nodes — a predictable workflow, not a
free-running multi-agent loop. Full write-up in [ARCHITECTURE.md](ARCHITECTURE.md).

![TriageFlow Lite architecture — nurse + Web UI, the LangGraph agent and its nodes, the open-weights model, the MCP client and both MCP servers with their tools, the deterministic SATS engine, audit log, data sources, the optional FHIR server, and the human approval gate, all inside an in-country/on-device boundary.](docs/img/architecture.png)

### MCP servers & tools

- **Own MCP server** (`triageflow`, stdio): `validate_triage_input`, `score_triage` ★,
  `check_deterioration` ★, `request_clinician_approval` ★ (the irreversible-action gate),
  `suggest_discriminators`, `fetch_fhir_vitals` (optional EMR import). ★ = action tools; every
  call is audit-logged. (Not a single-endpoint proxy.)
- **Borrowed MCP server**: the official
  [`@modelcontextprotocol/server-filesystem`](https://github.com/modelcontextprotocol/servers),
  read-only over `protocols/`, serving the cited SATS text. *Why borrow:* sandboxed,
  path-traversal-safe file access is a solved problem — reusing the reference is safer and
  faster than writing our own.
- **MCP client**: `langchain-mcp-adapters` `MultiServerMCPClient` launches both over stdio and
  loads their tools; the LangGraph nodes call them. Full catalog + call logic in [docs/MCP.md](docs/MCP.md).
- **Open-weights model** behind a pluggable, locally-managed runtime (llama.cpp or Ollama) —
  patient data never leaves the machine.

## Data

Synthetic only, and **FHIR R4-shaped** (see `src/triageflow/data/`): vitals carry **LOINC**
codes so an encounter looks like what a real EMR/FHIR API returns. Patients are drawn from
**26 clinical scenarios** written as a triage nurse would note them (*"C/O central chest pain
x1 hr, heavy and pressing, radiating to L arm. Sweating, nausea. Hx HTN, DM2. Pain 8/10."*),
each with vitals that **fit the story** — a patient in cardiac arrest has no desk vitals, an
asthmatic is breathing fast with a low SpO₂, numbers quoted in the note are the numbers in the
record. Every scenario is pinned by tests to its intended SATS colour. See
[docs/DATA.md](docs/DATA.md). We deliberately **do not use SNOMED CT** (its Affiliate licence
carries fees in non-member countries — an eligibility risk) or real datasets like MIMIC
(de-identified real data + out-of-country residency).

## Technology stack

- **Language:** Python 3.11+ (tested on 3.11 and 3.12, on Linux, Windows and macOS)
- **Agent / orchestration:** LangGraph (`StateGraph`, `MemorySaver` checkpointer, `interrupt()`)
- **LLM:** open-weights, local, behind an **OpenAI-compatible** client (`langchain-openai`) — pluggable backend: **llama.cpp** (`llama-server`, self-provisioned) or **Ollama**; models Qwen2.5 7B / 3B / 0.5B
- **MCP:** `mcp` (FastMCP) for our server · `langchain-mcp-adapters` (MultiServerMCPClient) client · borrowed `@modelcontextprotocol/server-filesystem` (npx)
- **Web:** FastAPI · Jinja2 · Tailwind · vanilla JS (SSE streaming) · Uvicorn
- **Data / validation:** Pydantic v2 · **FHIR R4** + **LOINC** · SQLite + JSONL (audit log)
- **External API (optional):** FHIR R4 server — in-country EMR, or HAPI sandbox for a demo
- **Triage scale:** South African Triage Scale (SATS) Adult / TEWS
- **Tooling:** `uv` · pytest (330 tests) · eval harness (`evals/`) · ruff · GitHub Actions on 3 operating systems · Apache-2.0

## Limitations

- **Adult pathway only** (older than 12 years / taller than 150 cm). A younger patient is
  refused with a pointer to the paediatric chart, never scored on the adult one.
- **Sign-finding depends on the model.** On 36 held-out lay phrasings the keyword map alone
  finds **14%** of the right signs; with the 7B model it is **86%** (see [EVALS.md](EVALS.md)).
  Suggestions never change a score until the nurse accepts them, and when the model has not
  read the complaint the nurse must confirm she checked the sign list — but a missed sign
  that she also misses is still an under-triage. This is the documented open failure (X1).
- **The complaint is read in English.** A few Swahili and Nigerian Pidgin phrases are in the
  keyword map as a start; they need review by local clinicians before being relied on.
- **Local conventions are not SATS.** The SpO₂ < 92% escalation and the 0–10 pain cut-offs
  are ours, are labelled on screen, and should be set by the facility's clinical lead.
- **The approver is a typed name**, not a login; there is no role check that an override
  really comes from a senior clinician. The waiting queue lives in memory for the session.
- **Speed on CPU.** The priority takes about 3 s; model suggestions a few seconds more once
  the model is loaded (about a minute after start-up, once).
- **No model-written handover note by default.** In testing, the 7B model turned a prompt
  ("do a finger-prick glucose test") into a claim ("glucose test done"). The plain-language
  summary is generated by code instead; the model's note is opt-in
  (`TRIAGEFLOW_HANDOVER_NOTE=1`), is not shown the actions, and is discarded if it claims
  anything was done.
- The optional FHIR import adds a network dependency; it is **off by default**.
- Not a medical device. Thresholds are a re-encoding of a published scale and must be
  clinically reviewed before any real use.

## Future improvements

- **Paediatric SATS** pathway (separate age-banded TEWS), and **GCS → AVPU** mapping.
- Close the sign-finding gap (X1): a constrained-decoding model pass, local-language
  phrasings reviewed by nurses, and recall tracked as a release metric.
- **Sign-in and roles** for the approval gate; a persistent, shared queue.
- A long-lived MCP session (each tool call currently starts a short-lived process).
- A borrowed **FHIR MCP** in front of an in-country EMR for pilots (see [docs/MCP.md](docs/MCP.md)).
- **Queue & flow** signals (staffing vs volume) and **referral routing** as further themes.
- Vendor the Tailwind/font CDN assets for fully-offline deployment.

## Licence & attribution

Code: **Apache-2.0** (see [LICENSE](LICENSE)). The SATS Adult rules are a fact-based
re-encoding of the **South African Triage Scale** (*SATS Training Manual 2012*, South
African Triage Group / EMSSA, CC BY-NC-SA 3.0); this repo ships only the re-encoded
thresholds with attribution, not the copyrighted charts. See [NOTICE](NOTICE).
