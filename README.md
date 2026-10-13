# TriageFlow Lite

**An agentic assistant for the busy OPD triage desk.** It turns a patient's vitals and
presenting complaint into a **sourced SATS priority level** in seconds — so the nurse can
see who needs to be looked at next, backed by the numbers already in front of her.

> **The one rule.** TriageFlow *scores and sequences* — it **never diagnoses, never
> prescribes, and never moves a patient**. A named nurse or clinician confirms every
> priority before it reaches a queue position. Every priority cites the vital or sign
> behind it; an unsourced flag is worse than no flag.

Theme: **Symptom-based priority scoring** · Protocol: **South African Triage Scale (SATS) Adult / TEWS**

![TriageFlow Lite — the nurse workspace: the waiting queue sorted by urgency, the agent's live tool calls across both MCP servers, and a RED priority with every point cited to the SATS chart.](docs/img/screenshot.png)

<p align="center"><sub>The nurse workspace — intake, the agent's live tool-call trace (own + borrowed MCP), and a sourced priority awaiting clinician confirmation.</sub></p>

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

Prerequisites: [`uv`](https://docs.astral.sh/uv/), [Node.js](https://nodejs.org) (for `npx`,
which runs the borrowed filesystem MCP — the official prebuilt installer is the most reliable
across platforms), and [Ollama](https://ollama.com/download) for the open-weights model. (If
`npx`/Node isn't available, the agent still runs — it falls back to reading the protocol
locally.)

```bash
uv run triageflow serve
```

Open <http://127.0.0.1:8000>. On first run `uv` installs everything; the app detects
Ollama and offers to download the model. Other commands:

```bash
uv run triageflow doctor        # check Ollama, model, and both MCP servers
uv run triageflow pull-model    # download the open-weights model (self-healing)
uv run triageflow demo          # run one assessment in the terminal, tool calls + approval
uv run python evals/run.py      # run the evaluation suite
uv run python -m pytest         # unit tests (pins the SATS/TEWS table)
```

**No GPU needed.** The clinical priority is computed by deterministic code (instant); the
open-weights model only extracts candidate signs from free text and writes the one-line
handover. Set `TRIAGEFLOW_NO_LLM=1` to run the agent fully deterministically (fast demos /
LLM-off sites).

### Models & backends (pick a lighter model, or swap the runtime)

The LLM runs behind an **OpenAI-compatible** interface, so the runtime is pluggable
(`TRIAGEFLOW_LLM_BACKEND`), and the agent code never changes.

- **Switch models from the UI** — the **model:** menu in the header lists a small catalog
  with download sizes; if your machine is low-powered, pick a **smaller** one (e.g. Qwen2.5
  **3B**) and it downloads + switches with live progress. Same from the terminal:
  `uv run triageflow pull-model qwen2.5:3b-instruct`.
- **`auto` (default)** — `uv run triageflow serve` detects what's already installed and uses
  it (**Ollama** or **llama.cpp**); only if neither is present does it set one up. No popups,
  no redundant setup. All downloads **resume + retry** across network interruptions.
- **Ollama** (zero-friction, best on macOS): downloads self-heal. Install from
  [ollama.com](https://ollama.com/download).
- **llama.cpp** (bare-metal, no daemon — good for constrained/offline boxes):
  `TRIAGEFLOW_LLM_BACKEND=llamacpp uv run triageflow serve`. GGUF models download from Hugging
  Face on demand (standard 7B, or a 3B / tiny 0.5B for low-end hardware); `llama-server` is
  launched + health-checked automatically.
  - **Windows:** the prebuilt `llama-server` is **auto-downloaded** on first run — nothing to install.
  - **Linux:** install `llama-server` (package or build), then run.
  - **macOS:** prefer **Ollama** (there is no official prebuilt llama.cpp binary; `brew install
    llama.cpp` builds from source and can take a long time, especially on Intel Macs).

`uv run triageflow doctor` shows the active backend, the resolved model, and the catalog.

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
| "We use **pulse rate**" | Relabelled Heart rate → **Pulse rate (HR)** |
| "**Pain score 0–10**" | Added pain 0–10, mapped to the **SATS pain discriminator** (4–6 → moderate/YELLOW, 7–10 → severe/ORANGE), sourced |
| "AVPU **or GCS**", "weight" | Added **weight**; kept AVPU (the SATS standard); GCS→AVPU mapping is noted as future work |

## What it does (the workflow it serves)

**Facility:** a public hospital outpatient department. **User:** the triage nurse at the
single desk that everyone passes through before they are assigned a priority.

She loads a patient (or picks one from the waiting queue), enters the presenting complaint
and the vitals she just measured, and clicks **Assess**. The agent then, visibly:

1. **validates** the input (missing / implausible / stale / contradictory / wrong-unit),
2. **extracts** candidate SATS signs from the free text — *for her to confirm*,
3. **retrieves** the cited protocol text from the borrowed filesystem MCP,
4. **scores** the SATS priority deterministically (TEWS + discriminators),
5. **checks itself** — confirms every point is cited and grounded in the protocol,
6. **pauses at the approval gate** — nothing is queued until she approves or overrides.

If a waiting patient is re-checked and has worsened, the agent raises a **deterioration
flag** for a second look — it still cannot move them.

The result is a priority a nurse can defend in a clinical audit: a colour band, a target
time, and every contributing vital/sign with its SATS citation — plus a readable, append-only
audit log of every tool call and the named approval.

## Human-in-the-loop

The approval gate is **structural, not a prompt instruction**: the agent is a LangGraph graph
that `interrupt()`s before its final step and **cannot reach `finalize` without a named
clinician's decision** supplied from the UI. The nurse/clinician **approves** the proposed
priority or **overrides** it (with a reason); only then is a confirmed priority placed in the
queue and written to the append-only audit log (who, what, when). A deterioration re-check only
ever **raises a flag for a second look** — the agent never moves a patient.

---

## Documentation

| Doc | What's in it |
|-----|--------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Agent shape, nodes, MCP built-vs-borrowed, control/cost/residency |
| [docs/MCP.md](docs/MCP.md) | The two MCP servers, full tool catalog + schemas, call logic |
| [docs/DATA.md](docs/DATA.md) | **Stakeholder data card** — how synthetic data is generated, explainability & trust, limitations |
| [EVALS.md](EVALS.md) | 10 test tasks, pass/fail, determinism, the documented unfixed failure |
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
- **Open-weights model** via locally-managed Ollama — patient data never leaves the machine.

## Data

Synthetic only, and **FHIR R4-shaped** (see `src/triageflow/data/`): vitals carry **LOINC**
codes so an encounter looks like what a real EMR/FHIR API returns. We deliberately **do not
use SNOMED CT** (its Affiliate licence carries fees in non-member countries — an eligibility
risk) or real datasets like MIMIC (de-identified real data + out-of-country residency). A
reproducible generator produces the waiting queue and a deterioration series.

## Technology stack

- **Language:** Python 3.11
- **Agent / orchestration:** LangGraph (`StateGraph`, `MemorySaver` checkpointer, `interrupt()`)
- **LLM:** open-weights, local, behind an **OpenAI-compatible** client (`langchain-openai`) — pluggable backend: **Ollama** (default) or **llama.cpp** (`llama-server`); models Qwen2.5 7B/3B, Llama-3.2-3B
- **MCP:** `mcp` (FastMCP) for our server · `langchain-mcp-adapters` (MultiServerMCPClient) client · borrowed `@modelcontextprotocol/server-filesystem` (npx)
- **Web:** FastAPI · Jinja2 · Tailwind · vanilla JS (SSE streaming) · Uvicorn
- **Data / validation:** Pydantic v2 · **FHIR R4** + **LOINC** · SQLite + JSONL (audit log)
- **External API (optional):** FHIR R4 server — in-country EMR, or HAPI sandbox for a demo
- **Triage scale:** South African Triage Scale (SATS) Adult / TEWS
- **Tooling:** `uv` · pytest (89 tests) · eval harness (`evals/`) · ruff · Apache-2.0

## Limitations

- Adult pathway only (>12y / >150cm). Paediatric SATS is out of scope for this version.
- The free-text sign extractor can miss or over-suggest; candidates **never** affect the
  score until the nurse confirms them (see the documented failure E8 in [EVALS.md](EVALS.md)).
- The optional FHIR import adds a network dependency; it is **off by default** and degrades
  gracefully. CPU LLM inference is slow on first load (the deterministic priority is instant).
- Not a medical device. Thresholds are a re-encoding of a published scale and must be
  clinically reviewed before any real use.

## Future improvements

- **Paediatric SATS** pathway (separate age-banded TEWS), and **GCS → AVPU** mapping.
- Shrink the extractor's under-triage gap (E8): synonym expansion done; next, a
  constrained-decoding LLM pass with a confidence threshold + a recall ("sensitivity") metric.
- A borrowed **FHIR MCP** in front of an in-country EMR for pilots (see [docs/MCP.md](docs/MCP.md)).
- **Queue & flow** signals (staffing vs volume) and **referral routing** as further themes.
- Vendor the Tailwind/font CDN assets for fully-offline deployment.

## Licence & attribution

Code: **Apache-2.0** (see [LICENSE](LICENSE)). The SATS Adult rules are a fact-based
re-encoding of the **South African Triage Scale** (*SATS Training Manual 2012*, South
African Triage Group / EMSSA, CC BY-NC-SA 3.0); this repo ships only the re-encoded
thresholds with attribution, not the copyrighted charts. See [NOTICE](NOTICE).
