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

The agent is a LangGraph state machine that, visibly, validates the input, extracts candidate
signs from free text *for the nurse to confirm*, retrieves the cited protocol text from a
borrowed filesystem MCP server, scores the priority **deterministically** via our own MCP
server, checks that every point is sourced and grounded, and then **pauses at an approval
gate**. Nothing reaches a queue position until a named clinician approves or overrides. If a
waiting patient is re-checked and has worsened, the agent raises a deterioration flag for a
second look — it never moves the patient itself.

The defensibility idea is a strict split: the **clinical score is pure deterministic code**
(zero run-to-run variance, a citation for every point), and the open-weights model
(Qwen2.5-7B via local Ollama, Llama-3.2-3B fallback) only extracts candidate signs and writes
the one-line handover. Every tool call and the named approval are written to an append-only
audit log, so a priority can survive a clinical-audit question.

Data is synthetic and FHIR R4-shaped (LOINC-coded vitals); no SNOMED CT (licensing) and no
real patient data. Patient data never leaves the machine. Runs in one command:
`uv run triageflow serve`.
