"""The TriageFlow agent — a LangGraph state machine with tool-calling nodes.

Shape (predictable workflow + agent nodes, per the architecture checklist — NOT a
free-running multi-agent loop):

    validate ─▶ extract ─▶ retrieve ─▶ score ─▶ deterioration ─▶ self_check ─▶ ⏸ approval ─▶ finalize
       │ (errors)
       └─────────────────────────────────────────────────────────────────────▶ END (needs correction)

  * Clinical scoring is deterministic (via the MCP ``score_triage`` tool). The LLM never
    decides a priority — it only extracts candidate signs from free text and writes a
    grounded explanation / self-check. Every LLM node has a deterministic fallback, so a
    slow or absent model degrades gracefully rather than breaking the run (recovery).
  * ``self_check`` verifies that the proposed priority's citations are grounded in the
    protocol text fetched from the *borrowed* filesystem MCP.
  * The graph ``interrupt()``s at the approval gate: nothing is confirmed until a named
    clinician approves or overrides. The web layer resumes with their decision.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from ..core import sats_rules as R
from ..core.audit import get_audit
from .mcp_client import TriageMCP

audit = get_audit()

_CATALOG = "\n".join(
    f"- {d.code}: {d.label} (implies {d.level.value})" for d in R.ALL_DISCRIMINATORS
)


class TriageState(TypedDict, total=False):
    encounter: dict[str, Any]
    recheck_vitals: dict[str, Any] | None
    nurse_confirmed_codes: list[str]
    model: str | None            # resolved Ollama model, or None for deterministic-only
    validation: dict[str, Any]
    candidates: list[dict[str, Any]]
    evidence_text: str
    result: dict[str, Any]
    deterioration: dict[str, Any] | None
    verification: dict[str, Any]
    explanation: str
    approval: dict[str, Any]
    status: str
    trace: Annotated[list[dict[str, Any]], operator.add]


def _t(step: str, server: str, summary: str, **extra: Any) -> dict[str, Any]:
    return {"step": step, "server": server, "summary": summary, **extra}


class TriageAgent:
    def __init__(self, mcp: TriageMCP, model: str | None = None, temperature: float = 0.0):
        self.mcp = mcp
        self.model = model
        self._llm = None
        if model:
            try:
                from langchain_ollama import ChatOllama
                # Cap generation + keep the model warm so narration stays snappy on CPU.
                # (The clinical priority itself is deterministic and instant — this LLM only
                #  refines candidate signs and writes the one-line handover.)
                self._llm = ChatOllama(model=model, temperature=temperature,
                                       num_predict=120, keep_alive="10m")
            except Exception:
                self._llm = None
        self.graph = self._build()

    # ---- nodes ---------------------------------------------------------------

    async def _validate(self, state: TriageState) -> dict[str, Any]:
        enc = state["encounter"]
        report = await self.mcp.call_json(
            "validate_triage_input",
            {"vitals": enc.get("vitals", {}), "encounter_id": enc.get("encounter_id", "ADHOC")})
        errs = [i for i in report.get("issues", []) if i.get("severity") == "error"]
        status = "needs_correction" if errs else "ok"
        return {"validation": report, "status": status,
                "trace": [_t("validate", "triageflow (own MCP)",
                             f"{len(report.get('issues', []))} issue(s); "
                             f"missing {report.get('missing_required', [])}", errors=len(errs))]}

    async def _extract(self, state: TriageState) -> dict[str, Any]:
        enc = state["encounter"]
        text = f"{enc.get('presenting_complaint','')} {enc.get('free_text_notes','')}".strip()
        sug = await self.mcp.call_json("suggest_discriminators", {"text": text})
        candidates = sug.get("candidates", [])
        source = "keyword map"
        # LLM refinement (optional): pick valid codes from the catalog, union with keywords.
        if self._llm and text:
            try:
                llm_codes = await self._llm_extract(text)
                valid = set(R.DISCRIMINATORS_BY_CODE)
                seen = {c["code"] for c in candidates}
                for code in llm_codes:
                    if code in valid and code not in seen:
                        d = R.DISCRIMINATORS_BY_CODE[code]
                        candidates.append({"code": code, "label": d.label,
                                           "implies": d.level.value, "source": d.source,
                                           "via": "llm"})
                source = f"keyword map + {self.model}"
            except Exception as e:
                return {"candidates": candidates,
                        "trace": [_t("extract", "agent",
                                     f"LLM extract unavailable ({type(e).__name__}); "
                                     f"used keyword map", recovered=True)]}
        return {"candidates": candidates,
                "trace": [_t("extract", "agent",
                             f"{len(candidates)} candidate sign(s) via {source} "
                             f"(nurse confirms)")]}

    async def _llm_extract(self, text: str) -> list[str]:
        from langchain_core.messages import HumanMessage, SystemMessage
        sys = SystemMessage(content=(
            "You map a triage complaint to SATS discriminator CODES. Only choose from this "
            "catalog; never invent codes. Reply with a comma-separated list of codes, or 'none'.\n"
            + _CATALOG))
        msg = HumanMessage(content=f"Complaint: {text}\nCodes:")
        resp = await self._llm.ainvoke([sys, msg])
        raw = (resp.content or "").strip().lower()
        if "none" in raw:
            return []
        return [c.strip() for c in raw.replace("\n", ",").split(",") if c.strip()]

    async def _retrieve(self, state: TriageState) -> dict[str, Any]:
        text = await self.mcp.read_protocol("sats_adult.md")
        via = "filesystem (borrowed MCP)" if self.mcp.has("read_text_file") else "local fallback"
        return {"evidence_text": text,
                "trace": [_t("retrieve", via, f"fetched protocol evidence ({len(text)} chars)")]}

    async def _score(self, state: TriageState) -> dict[str, Any]:
        enc = dict(state["encounter"])
        codes = list(dict.fromkeys(
            list(enc.get("discriminator_codes", [])) + list(state.get("nurse_confirmed_codes", []))))
        enc["discriminator_codes"] = codes
        result = await self.mcp.call_json("score_triage", {"encounter": enc})
        pr = result.get("proposed_priority")
        return {"result": result,
                "trace": [_t("score", "triageflow (own MCP)",
                             f"proposed {pr}; TEWS {result.get('tews_total')}; "
                             f"{len([e for e in result.get('evidence',[]) if e.get('source')])} citations",
                             deterministic=True)]}

    async def _deterioration(self, state: TriageState) -> dict[str, Any]:
        recheck = state.get("recheck_vitals")
        if not recheck:
            return {"deterioration": None}
        enc = state["encounter"]
        out = await self.mcp.call_json("check_deterioration", {
            "encounter_id": enc.get("encounter_id", "ADHOC"),
            "baseline_vitals": enc.get("vitals", {}), "recheck_vitals": recheck})
        return {"deterioration": out,
                "trace": [_t("deterioration", "triageflow (own MCP)",
                             f"escalated={out.get('escalated')} (ΔTEWS {out.get('delta')})")]}

    async def _self_check(self, state: TriageState) -> dict[str, Any]:
        result = state.get("result", {})
        evidence = result.get("evidence", [])
        proto = state.get("evidence_text", "")
        # Deterministic grounding: every evidence item is cited, and its governing rule
        # concept appears in the retrieved protocol text.
        uncited = [e for e in evidence if not e.get("source")]
        grounded = True
        notes = []
        for e in evidence:
            label = (e.get("band_label") or e.get("observed") or "")
            if e.get("kind") == "discriminator" and label and label.lower() not in proto.lower():
                grounded = False
                notes.append(f"'{label}' not found verbatim in protocol")
        if "TEWS" not in proto:
            grounded = False
            notes.append("TEWS table not found in retrieved protocol")
        verification = {
            "all_cited": not uncited,
            "grounded_in_protocol": grounded,
            "notes": notes,
            "citation_count": len([e for e in evidence if e.get("source")]),
        }
        explanation = await self._explain(state, verification)
        ok = verification["all_cited"] and grounded
        return {"verification": verification, "explanation": explanation,
                "trace": [_t("self_check", "agent",
                             f"citations {'all present' if not uncited else 'MISSING'}; "
                             f"grounded={grounded}", passed=ok)]}

    async def _explain(self, state: TriageState, verification: dict[str, Any]) -> str:
        result = state.get("result", {})
        pr = result.get("proposed_priority")
        if not result.get("complete"):
            base = ("Assessment incomplete — required vitals missing and no overriding sign. "
                    f"Needs: {', '.join(result.get('missing', []))}.")
        else:
            drivers = []
            for e in result.get("evidence", []):
                if e.get("kind") == "discriminator":
                    drivers.append(f"{e.get('observed')} (→{e.get('implies')})")
                elif e.get("points"):
                    drivers.append(f"{e.get('parameter')} {e.get('observed')} (+{e.get('points')})")
            base = (f"Proposed {pr} — target {result.get('target_time')}. "
                    f"TEWS {result.get('tews_total')}. Key drivers: "
                    f"{'; '.join(drivers[:4]) or 'all vitals normal'}. "
                    f"Awaiting clinician confirmation.")
        if self._llm:
            try:
                from langchain_core.messages import HumanMessage, SystemMessage
                sys = SystemMessage(content=(
                    "You are a triage assistant writing a ONE-sentence, factual handover for a "
                    "nurse. Use ONLY the evidence given. Do NOT diagnose or suggest treatment. "
                    "State the proposed priority and the vitals/signs behind it."))
                msg = HumanMessage(content=f"Evidence JSON: {result.get('evidence')}\n"
                                           f"Proposed: {pr}, TEWS {result.get('tews_total')}.")
                resp = await self._llm.ainvoke([sys, msg])
                if resp.content and resp.content.strip():
                    return resp.content.strip()
            except Exception:
                return base
        return base

    async def _approval(self, state: TriageState) -> dict[str, Any]:
        result = state.get("result", {})
        payload = {
            "encounter_id": state["encounter"].get("encounter_id"),
            "proposed_priority": result.get("proposed_priority"),
            "explanation": state.get("explanation"),
            "verification": state.get("verification"),
        }
        audit.record("await_approval", encounter_id=payload["encounter_id"],
                     tool="agent", payload={"proposed": payload["proposed_priority"]})
        decision = interrupt(payload)  # ⏸ pause for the human gate
        return {"approval": decision, "status": "approved"}

    async def _finalize(self, state: TriageState) -> dict[str, Any]:
        decision = state.get("approval", {}) or {}
        result = state.get("result", {})
        confirmed = await self.mcp.call_json("request_clinician_approval", {
            "encounter_id": state["encounter"].get("encounter_id"),
            "proposed_priority": result.get("proposed_priority"),
            "clinician_name": decision.get("clinician_name", "unknown"),
            "decision": decision.get("decision", "approve"),
            "override_priority": decision.get("override_priority"),
            "reason": decision.get("reason", ""),
        })
        return {"approval": {**decision, "confirmed": confirmed}, "status": "finalized",
                "trace": [_t("finalize", "triageflow (own MCP)",
                             f"{decision.get('decision','approve')} by "
                             f"{decision.get('clinician_name','?')} → "
                             f"{confirmed.get('confirmed_priority')}", gate=True)]}

    # ---- wiring --------------------------------------------------------------

    def _route_after_validate(self, state: TriageState) -> str:
        return END if state.get("status") == "needs_correction" else "extract"

    def _build(self):
        g = StateGraph(TriageState)
        g.add_node("validate", self._validate)
        g.add_node("extract", self._extract)
        g.add_node("retrieve", self._retrieve)
        g.add_node("score", self._score)
        g.add_node("deterioration", self._deterioration)
        g.add_node("self_check", self._self_check)
        g.add_node("approval", self._approval)
        g.add_node("finalize", self._finalize)

        g.add_edge(START, "validate")
        g.add_conditional_edges("validate", self._route_after_validate,
                                {"extract": "extract", END: END})
        g.add_edge("extract", "retrieve")
        g.add_edge("retrieve", "score")
        g.add_edge("score", "deterioration")
        g.add_edge("deterioration", "self_check")
        g.add_edge("self_check", "approval")
        g.add_edge("approval", "finalize")
        g.add_edge("finalize", END)
        return g.compile(checkpointer=MemorySaver())

    # ---- public API ----------------------------------------------------------

    async def start(self, encounter: dict[str, Any], *, recheck_vitals=None,
                    nurse_confirmed_codes=None) -> dict[str, Any]:
        """Run to the approval gate. Returns the state incl. the interrupt payload."""
        config = {"configurable": {"thread_id": encounter.get("encounter_id", "ADHOC")}}
        init: TriageState = {
            "encounter": encounter, "recheck_vitals": recheck_vitals,
            "nurse_confirmed_codes": nurse_confirmed_codes or [],
            "model": self.model, "trace": [],
        }
        state = await self.graph.ainvoke(init, config)
        return state

    async def astream_start(self, encounter: dict[str, Any], *, recheck_vitals=None,
                            nurse_confirmed_codes=None):
        """Async generator yielding live events as each node completes, for the UI.

        Yields dicts: {"type": "trace", "entry": {...}} per node, then
        {"type": "proposal", "state": {...}} when the graph pauses at the approval gate.
        """
        config = {"configurable": {"thread_id": encounter.get("encounter_id", "ADHOC")}}
        init: TriageState = {
            "encounter": encounter, "recheck_vitals": recheck_vitals,
            "nurse_confirmed_codes": nurse_confirmed_codes or [],
            "model": self.model, "trace": [],
        }
        async for chunk in self.graph.astream(init, config, stream_mode="updates"):
            if "__interrupt__" in chunk:
                continue
            for _node, update in chunk.items():
                if not isinstance(update, dict):
                    continue
                for entry in update.get("trace", []) or []:
                    yield {"type": "trace", "entry": entry}
                if update.get("candidates") is not None:
                    yield {"type": "candidates", "candidates": update["candidates"]}
        state = await self.graph.aget_state(config)
        yield {"type": "proposal", "state": _public_state(state.values)}

    async def approve(self, encounter_id: str, decision: dict[str, Any]) -> dict[str, Any]:
        """Resume the paused graph with the clinician's decision, run finalize → END."""
        config = {"configurable": {"thread_id": encounter_id}}
        state = await self.graph.ainvoke(Command(resume=decision), config)
        return state


def _public_state(values: dict[str, Any]) -> dict[str, Any]:
    """Trim internal channels to what the UI needs."""
    keys = ("result", "verification", "explanation", "candidates", "validation",
            "deterioration", "status", "trace")
    return {k: values.get(k) for k in keys}
