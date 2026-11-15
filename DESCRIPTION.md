# Submission description

**Sub-theme:** Symptom-based priority scoring.
**Facility & workflow:** the single triage desk of a public-hospital outpatient department
(OPD), used by the triage nurse who assigns a priority to everyone before they join a bay
queue.

TriageFlow Lite gives that nurse an agent that turns the vitals and presenting complaint
already in front of her into a **sourced SATS priority level** in seconds — so the wait
itself stops being where the risk hides. It is built on the **South African Triage Scale
(Adult / TEWS)**: an African, physiologically-scored scale whose per-vital points make every
priority citable.

The agent is a LangGraph state machine that, visibly, validates the input, retrieves the
protocol text from a borrowed filesystem MCP server, scores the priority **deterministically**
via our own MCP server — on screen within seconds, with the reasons in plain words and the
chart's "do now" checks — verifies every point against the protocol, suggests signs from the
free-text complaint *for the nurse to accept or dismiss*, and then **pauses at an approval
gate**. Nothing reaches the queue until a named clinician confirms; the gate refuses an
incomplete result, an unreviewed sign, or an override without a reason. If a waiting patient
is re-checked and has worsened, the agent raises a flag for a second look — it never moves
the patient itself.

The defensibility idea is a strict split: the **clinical score is pure deterministic code**
(zero run-to-run variance, a citation for every point), and the open-weights model
(Qwen2.5 via local llama.cpp, or Ollama if already installed) only suggests signs for the
nurse to review. Every tool call and the named approval are written to an append-only audit
log, so a priority can survive a clinical-audit question.

Data is synthetic and FHIR R4-shaped (LOINC-coded vitals); no SNOMED CT (licensing) and no
real patient data. Patient data never leaves the machine. Runs in one command:
`uv run triageflow serve`.
