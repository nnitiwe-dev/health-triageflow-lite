# Demo script (< 3 minutes, unedited agent run)

Goal: show the agent *working*, with real tool calls on screen, including where it fails.
Record the terminal + browser. Keep it to one take.

**Setup (before recording):** `uv run triageflow pull-model` once so the model is warm.
For a snappy run you may export `TRIAGEFLOW_NO_LLM=1` (deterministic) — say so on camera if
you do; the priority is identical either way.

---

### 0:00 — What & the hard rule (15s)
"TriageFlow Lite helps the OPD triage nurse. It *scores and sequences* — it never diagnoses,
never prescribes, never moves a patient. Every priority cites its source, and a clinician
confirms it." Point to the `⚠ not a diagnosis` chip and the `MCP: triageflow + filesystem`
chip.

### 0:15 — The queue (15s)
"This is the waiting queue, sorted by urgency — chest pain and an active seizure float to the
top, routine checks sink." Click the **chest-pain** patient to load it.

### 0:30 — Run the agent, narrate the tool calls (45s)
Click **Assess**. As the trace streams, call out each step:
- `validate` → own MCP server checks the vitals;
- `extract` → candidate signs surface for the nurse to **confirm** (not auto-applied);
- `retrieve` → the **borrowed** filesystem MCP fetches the cited SATS text;
- `score` → our MCP server computes the priority **deterministically**;
- `self_check` → confirms every point is sourced and grounded.

### 1:15 — The sourced proposal (30s)
"Proposed **RED**, target Immediate. Look at the evidence — every vital with its TEWS points
and its SATS citation, plus the chest-pain discriminator. Badges: all sourced, grounded in
protocol, N citations."

### 1:45 — The human gate (25s)
Type a clinician name, click **Confirm priority & place in queue**. "Nothing moved until a
named clinician approved." Open **view audit trail**: show `validate → score → await_approval
→ approval` with timestamps and the approver's name.

### 2:10 — Deterioration (20s)
Load the deterioration example (`/api/demo/deterioration`) or re-check a waiting patient:
"On re-check the TEWS jumped — the agent raises a second-look flag. It asks a human to
re-look; it cannot move the patient."

### 2:30 — Where it fails (honest) (25s)
Type a red-flag complaint in *free text only* with normal vitals — e.g. "crushing
retrosternal discomfort radiating to jaw" — and no confirmed sign. "It under-triages to
GREEN, because we never auto-apply an unconfirmed sign. That's eval case E8, documented in
EVALS.md, with the fix we'd try next." End.

---

### Terminal-only alternative
`uv run triageflow demo` prints the full tool-call trace, the sourced priority, the pause at
the gate, and the confirmed result — good for a split-screen or a fallback.
