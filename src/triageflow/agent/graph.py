"""The TriageFlow agent — a LangGraph state machine with tool-calling nodes.

Shape (predictable workflow + agent nodes, per the architecture checklist — NOT a
free-running multi-agent loop):

    validate ─▶ retrieve ─▶ score ─▶ deterioration ─▶ self_check ─▶ extract ─▶ ⏸ approval ─▶ finalize
       │ (impossible values, no emergency sign)
       └──────────────────────────────────────────────────────────────────────▶ END (needs correction)

  * Clinical scoring is deterministic (via the MCP ``score_triage`` tool) and comes FIRST, so
    the priority is on screen in about a second. The LLM never decides a priority and never
    delays one: it only suggests candidate signs from free text afterwards (``extract``), and
    can write an optional handover note outside the gate (``narrate``).
  * Every LLM step has a deterministic fallback and a timeout, so a slow or absent model
    degrades gracefully rather than breaking the run (recovery).
  * ``self_check`` verifies each piece of evidence against the protocol text fetched from the
    *borrowed* filesystem MCP — the right band in the right TEWS column, each sign in the
    section for its level — and reports any rule that is a local convention rather than SATS.
  * The graph ``interrupt()``s at the approval gate: nothing is confirmed until a named
    clinician approves or overrides. ``approve()`` enforces the gate rules on the server:
    no approving an incomplete assessment, no approving while suggested signs are unreviewed,
    no approving without a sign-list check when the model did not read the complaint, and
    no override without a reason.
  * Every run gets its own thread (``run_id``), so a result can never be a leftover from a
    previous assessment of the same encounter.
"""

from __future__ import annotations

import asyncio
import operator
import re
import uuid
from collections import deque
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from ..core import sats_rules as R
from ..core.audit import get_audit
from ..core.explain import build_summary, format_text
from .mcp_client import TriageMCP

audit = get_audit()

_CATALOG = "\n".join(
    f"- {d.code}: {d.label} (implies {d.level.value})" for d in R.ALL_DISCRIMINATORS
)
_VALID_PRIORITIES = {p.value for p in R.Priority}
_CODE_TOKEN = re.compile(r"[a-z][a-z0-9_]+")
# Words by which a handover note would claim (or order) a test or treatment.
_CLAIMS_ACTION = re.compile(
    r"\b(done|completed|performed|given|administered|started|commenced|ordered|pending|"
    r"taken|obtained|prescribed?)\b", re.IGNORECASE)


class ApprovalError(Exception):
    """The approval gate refused the decision (the message is safe to show the clinician)."""


class TriageState(TypedDict, total=False):
    run_id: str
    encounter: dict[str, Any]
    recheck_vitals: dict[str, Any] | None      # legacy: compare encounter vitals -> these
    baseline_vitals: dict[str, Any] | None     # re-check flow: compare these -> encounter vitals
    nurse_confirmed_codes: list[str]
    nurse_dismissed_codes: list[str]
    prior_candidates: list[dict[str, Any]] | None  # reuse (skip extraction) on a re-score
    prior_extraction: dict[str, Any] | None        # how those candidates were found
    model: str | None            # resolved model, or None for deterministic-only
    validation: dict[str, Any]
    candidates: list[dict[str, Any]]
    extraction: dict[str, Any]
    evidence_text: str
    result: dict[str, Any]
    summary: dict[str, Any]
    deterioration: dict[str, Any] | None
    verification: dict[str, Any]
    explanation: str
    approval: dict[str, Any]
    status: str
    trace: Annotated[list[dict[str, Any]], operator.add]


def _t(step: str, server: str, summary: str, **extra: Any) -> dict[str, Any]:
    return {"step": step, "server": server, "summary": summary, **extra}


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").lower().replace("–", "-").replace("—", "-"))


def _protocol_section(proto: str, heading_prefix: str) -> str:
    """Text of the markdown section whose heading starts with ``heading_prefix``."""
    m = re.search(rf"^#+\s*{re.escape(heading_prefix)}.*?$", proto, re.MULTILINE)
    if not m:
        return ""
    nxt = re.search(r"^#+\s", proto[m.end():], re.MULTILINE)
    return proto[m.end(): m.end() + nxt.start()] if nxt else proto[m.end():]


_TEWS_ROW = {"respiratory_rate": "Respiratory rate", "heart_rate": "Heart rate",
             "systolic_bp": "Systolic BP", "temperature_c": "Temperature", "avpu": "AVPU",
             "mobility": "Mobility", "trauma": "Trauma"}
_TEWS_COL_POINTS = [3, 2, 1, 0, 1, 2, 3]
_SIGN_SECTION = {"RED": "5.1", "ORANGE": "5.2", "YELLOW": "5.3"}


def verify_against_protocol(evidence: list[dict[str, Any]], proto: str) -> dict[str, Any]:
    """Check every evidence item against the protocol text.

    * TEWS: the band must sit in the protocol table's row for that parameter, in the column
      whose points equal the points awarded.
    * Signs: the label must be listed in the protocol section for the level it implies.
    * Local conventions (e.g. the SpO2 safety floor) are NOT SATS: they are reported in
      ``local_rules`` so the UI can say so, rather than being passed as "grounded".
    """
    notes: list[str] = []
    local_rules: list[str] = []
    uncited = [e for e in evidence if not e.get("source")]
    table = _protocol_section(proto, "4. Triage Early Warning Score")
    rows = {}
    for line in table.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 8:
            rows[_norm(cells[0]).split("(")[0]] = cells[1:]
    if not rows:
        notes.append("TEWS table not found in the retrieved protocol")

    for e in evidence:
        kind, label = e.get("kind"), e.get("band_label") or ""
        if kind == "tews" and rows:
            row = rows.get(_norm(_TEWS_ROW.get(e.get("parameter"), "")))
            ok = bool(row) and any(
                _norm(label) in _norm(cell) and _TEWS_COL_POINTS[i] == (e.get("points") or 0)
                for i, cell in enumerate(row))
            if not ok:
                notes.append(f"TEWS band '{label}' (+{e.get('points')}) for {e.get('parameter')} "
                             "does not match the protocol table")
        elif kind == "discriminator":
            section = _protocol_section(proto, _SIGN_SECTION.get(e.get("implies"), "5."))
            if _norm(label) not in _norm(section):
                notes.append(f"Sign '{label}' is not listed under {e.get('implies')} in the protocol")
        elif kind == "additional_investigation":
            if _norm("3 mmol/L or less") not in _norm(_protocol_section(proto, "6.")):
                notes.append("Hypoglycaemia threshold not found in the protocol")
        elif kind == "safety_rule":
            local_rules.append(label)
    return {
        "all_cited": not uncited,
        "grounded_in_protocol": not notes,
        "notes": notes,
        "local_rules": local_rules,
        "citation_count": len([e for e in evidence if e.get("source")]),
    }


class TriageAgent:
    def __init__(self, mcp: TriageMCP, model: str | None = None, temperature: float = 0.0,
                 chat_base_url: str | None = None, extract_timeout: float = 25.0):
        self.mcp = mcp
        self.model = model
        self.extract_timeout = extract_timeout
        self._llm = None
        self._runs: deque[str] = deque(maxlen=300)
        # Backend-agnostic: talk to whatever OpenAI-compatible server (llama.cpp / Ollama /
        # vLLM) the chosen backend exposes. model=None => deterministic-only (no LLM calls).
        if model and chat_base_url:
            try:
                from .llm_backend import make_chat_model
                self._llm = make_chat_model(chat_base_url, model, temperature=temperature,
                                            max_tokens=120)
            except Exception:
                self._llm = None
        self.graph = self._build()

    @property
    def llm_enabled(self) -> bool:
        return self._llm is not None

    # ---- nodes ---------------------------------------------------------------

    def _confirmed(self, state: TriageState) -> list[str]:
        enc = state["encounter"]
        return list(dict.fromkeys(
            list(enc.get("discriminator_codes", [])) + list(state.get("nurse_confirmed_codes", []))))

    async def _validate(self, state: TriageState) -> dict[str, Any]:
        enc = dict(state["encounter"])
        report = await self.mcp.call_json(
            "validate_triage_input",
            {"vitals": enc.get("vitals", {}), "encounter_id": enc.get("encounter_id", "ADHOC"),
             "age_years": enc.get("age_years")})
        errs = [i for i in report.get("issues", []) if i.get("severity") == "error"]
        warns = [i for i in report.get("issues", []) if i.get("severity") == "warning"]
        # SATS step 1: an emergency sign means RED with no delay — a mistyped vital must not
        # hold that up. Drop the impossible values (they become "missing") and carry on.
        emergency = any(R.DISCRIMINATORS_BY_CODE[c].level == R.Priority.RED
                        for c in self._confirmed(state) if c in R.DISCRIMINATORS_BY_CODE)
        out: dict[str, Any] = {"validation": report}
        if errs and emergency:
            bad = {i.get("field") for i in errs}
            enc["vitals"] = {k: v for k, v in (enc.get("vitals") or {}).items() if k not in bad}
            out["encounter"] = enc
            status, note = "ok", " — emergency sign confirmed, so scoring continues without them"
        elif errs:
            status, note = "needs_correction", " — correct these before a priority can be given"
        else:
            status, note = "ok", ""
        out["status"] = status
        out["trace"] = [_t("validate", "triageflow (own MCP)",
                           f"{len(errs)} error(s), {len(warns)} warning(s); "
                           f"missing {report.get('missing_required', [])}{note}", errors=len(errs))]
        return out

    async def _retrieve(self, state: TriageState) -> dict[str, Any]:
        text = await self.mcp.read_protocol("sats_adult.md")
        via = getattr(self.mcp, "protocol_source", "local file")
        return {"evidence_text": text,
                "trace": [_t("retrieve", via, f"fetched protocol evidence ({len(text)} chars)")]}

    async def _score(self, state: TriageState) -> dict[str, Any]:
        enc = dict(state["encounter"])
        enc["discriminator_codes"] = self._confirmed(state)
        result = await self.mcp.call_json("score_triage", {"encounter": enc})
        summary = build_summary(result)
        # Data-quality warnings belong next to the result, in words, where the nurse reads.
        for issue in (state.get("validation") or {}).get("issues", []):
            if issue.get("message") not in summary["cautions"]:
                summary["cautions"].append(issue.get("message"))
        pr = result.get("proposed_priority")
        return {"result": result, "summary": summary, "explanation": summary["sentence"],
                "trace": [_t("score", "triageflow (own MCP)",
                             f"proposed {pr}; TEWS {result.get('tews_total')}; "
                             f"{len([e for e in result.get('evidence', []) if e.get('source')])} "
                             f"citations; {len(result.get('actions', []))} action(s)",
                             deterministic=True)]}

    async def _deterioration(self, state: TriageState) -> dict[str, Any]:
        enc = state["encounter"]
        baseline, recheck = state.get("baseline_vitals"), state.get("recheck_vitals")
        if baseline:                       # re-check flow: baseline -> what was just entered
            args = {"baseline_vitals": baseline, "recheck_vitals": enc.get("vitals", {})}
        elif recheck:                      # legacy: what was entered -> a later re-check
            args = {"baseline_vitals": enc.get("vitals", {}), "recheck_vitals": recheck}
        else:
            return {"deterioration": None}
        out = await self.mcp.call_json("check_deterioration", {
            "encounter_id": enc.get("encounter_id", "ADHOC"), **args})
        return {"deterioration": out,
                "trace": [_t("deterioration", "triageflow (own MCP)",
                             f"escalated={out.get('escalated')} (ΔTEWS {out.get('delta')})"
                             + (f"; re-check missing {out.get('missing')}"
                                if out.get("missing") else ""))]}

    async def _self_check(self, state: TriageState) -> dict[str, Any]:
        result = state.get("result", {})
        verification = verify_against_protocol(result.get("evidence", []),
                                               state.get("evidence_text", ""))
        ok = verification["all_cited"] and verification["grounded_in_protocol"]
        detail = "; ".join(verification["notes"][:2])
        return {"verification": verification,
                "trace": [_t("self_check", "agent",
                             f"citations {'all present' if verification['all_cited'] else 'MISSING'}; "
                             f"matches protocol={verification['grounded_in_protocol']}"
                             + (f" ({detail})" if detail else "")
                             + (f"; {len(verification['local_rules'])} local rule(s)"
                                if verification["local_rules"] else ""), passed=ok)]}

    def _mark(self, candidates: list[dict[str, Any]], state: TriageState) -> list[dict[str, Any]]:
        """Attach each candidate's review status: present / absent / unreviewed."""
        confirmed = set(self._confirmed(state))
        dismissed = set(state.get("nurse_dismissed_codes", []))
        derived = {e.get("parameter") for e in (state.get("result") or {}).get("evidence", [])
                   if e.get("kind") == "discriminator"} - confirmed
        out = []
        for c in candidates:
            code = c["code"]
            status = ("present" if code in confirmed else
                      "from_vitals" if code in derived else
                      "absent" if code in dismissed else "unreviewed")
            out.append({**c, "status": status})
        return out

    @staticmethod
    def _vitals_candidates(enc: dict[str, Any]) -> list[dict[str, Any]]:
        """Signs a structured reading makes worth asking about. A high glucose only becomes a
        SATS sign if the patient is diabetic (and depends on ketones) — the nurse knows that,
        the number alone does not — so it is offered as a suggestion, never auto-applied."""
        g = (enc.get("vitals") or {}).get("glucose_mmol")
        codes = []
        if isinstance(g, int | float) and g > R.HYPERGLYCAEMIA_MIN_MMOL:
            codes.append("diabetic_gt11_ketonuria")
            if g > 17:
                codes.append("diabetic_gt17")
        return [{"code": c, "label": R.DISCRIMINATORS_BY_CODE[c].label,
                 "implies": R.DISCRIMINATORS_BY_CODE[c].level.value,
                 "source": R.DISCRIMINATORS_BY_CODE[c].source, "via": "vitals"} for c in codes]

    async def _extract(self, state: TriageState) -> dict[str, Any]:
        enc = state["encounter"]
        prior = state.get("prior_candidates")
        from_vitals = self._vitals_candidates(enc)
        if prior is not None:   # re-score after the nurse's review: nothing new to read
            prior = [c for c in prior if c.get("via") != "vitals"]
            prior += [c for c in from_vitals if c["code"] not in {p["code"] for p in prior}]
            was = state.get("prior_extraction") or {}
            return {"candidates": self._mark(prior, state),
                    "extraction": {"mode": was.get("mode", "keywords"),
                                   "note": was.get("note", ""), "reused": True},
                    "trace": [_t("extract", "agent", "re-used the reviewed candidate signs")]}

        text = f"{enc.get('presenting_complaint', '')} {enc.get('free_text_notes', '')}".strip()
        sug = await self.mcp.call_json("suggest_discriminators", {"text": text})
        candidates = [{**c, "via": "keywords"} for c in sug.get("candidates", [])]
        candidates += [c for c in from_vitals if c["code"] not in {x["code"] for x in candidates}]
        mode, note, recovered = "keywords", "", False
        if not text:
            note = "No presenting complaint entered — use the sign checklist."
        elif self._llm:
            try:
                llm_codes = await asyncio.wait_for(self._llm_extract(text), self.extract_timeout)
                seen = {c["code"] for c in candidates}
                for code in llm_codes:
                    if code not in seen:
                        d = R.DISCRIMINATORS_BY_CODE[code]
                        candidates.append({"code": code, "label": d.label,
                                           "implies": d.level.value, "source": d.source,
                                           "via": "model"})
                        seen.add(code)
                mode = "model+keywords"
            except Exception as e:
                recovered = True
                why = "took too long" if isinstance(e, TimeoutError) else "was unavailable"
                note = (f"The language model {why}, so only keyword matching was used. "
                        "Check the full sign list for anything it may have missed.")
        else:
            note = ("Language model is off — only keyword matching was used. "
                    "Check the full sign list for anything it may have missed.")
        how = {"keywords": "keyword map", "model+keywords": f"keyword map + {self.model}"}[mode]
        return {"candidates": self._mark(candidates, state),
                "extraction": {"mode": mode, "note": note},
                "trace": [_t("extract", "agent",
                             f"{len(candidates)} candidate sign(s) via {how} (nurse confirms)"
                             + (" — model fallback" if recovered else ""), recovered=recovered)]}

    async def _llm_extract(self, text: str) -> list[str]:
        from langchain_core.messages import HumanMessage, SystemMessage
        sys = SystemMessage(content=(
            "You map a triage complaint to SATS discriminator CODES. Only choose from this "
            "catalog; never invent codes. Ignore findings the note denies (e.g. 'no chest "
            "pain'). Reply with a comma-separated list of codes, or 'none'.\n" + _CATALOG))
        msg = HumanMessage(content=f"Complaint: {text}\nCodes:")
        resp = await self._llm.ainvoke([sys, msg])
        raw = (resp.content or "").strip().lower()
        # Robust to formatting: keep only tokens that are real catalog codes.
        return list(dict.fromkeys(
            t for t in _CODE_TOKEN.findall(raw) if t in R.DISCRIMINATORS_BY_CODE))

    async def _approval(self, state: TriageState) -> dict[str, Any]:
        result = state.get("result", {})
        payload = {
            "encounter_id": state["encounter"].get("encounter_id"),
            "proposed_priority": result.get("proposed_priority"),
            "explanation": state.get("explanation"),
            "verification": state.get("verification"),
        }
        audit.record("await_approval", encounter_id=payload["encounter_id"],
                     tool="agent", payload={"proposed": payload["proposed_priority"],
                                            "run_id": state.get("run_id")})
        decision = interrupt(payload)  # ⏸ pause for the human gate
        return {"approval": decision, "status": "approved"}

    async def _finalize(self, state: TriageState) -> dict[str, Any]:
        decision = state.get("approval", {}) or {}
        result = state.get("result", {})
        confirmed = await self.mcp.call_json("request_clinician_approval", {
            "encounter_id": state["encounter"].get("encounter_id"),
            "proposed_priority": result.get("proposed_priority"),
            "clinician_name": decision.get("clinician_name", ""),
            "decision": decision.get("decision", "approve"),
            "override_priority": decision.get("override_priority"),
            "reason": decision.get("reason", ""),
        })
        return {"approval": {**decision, "confirmed": confirmed}, "status": "finalized",
                "trace": [_t("finalize", "triageflow (own MCP)",
                             f"{decision.get('decision', 'approve')} by "
                             f"{decision.get('clinician_name', '?')} → "
                             f"{confirmed.get('confirmed_priority')}", gate=True)]}

    # ---- wiring --------------------------------------------------------------

    def _route_after_validate(self, state: TriageState) -> str:
        return END if state.get("status") == "needs_correction" else "retrieve"

    def _build(self):
        g = StateGraph(TriageState)
        g.add_node("validate", self._validate)
        g.add_node("retrieve", self._retrieve)
        g.add_node("score", self._score)
        g.add_node("deterioration", self._deterioration)
        g.add_node("self_check", self._self_check)
        g.add_node("extract", self._extract)
        g.add_node("approval", self._approval)
        g.add_node("finalize", self._finalize)

        g.add_edge(START, "validate")
        g.add_conditional_edges("validate", self._route_after_validate,
                                {"retrieve": "retrieve", END: END})
        g.add_edge("retrieve", "score")
        g.add_edge("score", "deterioration")
        g.add_edge("deterioration", "self_check")
        g.add_edge("self_check", "extract")
        g.add_edge("extract", "approval")
        g.add_edge("approval", "finalize")
        g.add_edge("finalize", END)
        return g.compile(checkpointer=MemorySaver())

    # ---- public API ----------------------------------------------------------

    def _new_run(self, encounter: dict[str, Any], **kw: Any) -> tuple[str, dict, TriageState]:
        """A fresh thread per run: state from an earlier assessment can never leak in."""
        run_id = f"{encounter.get('encounter_id', 'ADHOC')}::{uuid.uuid4().hex[:10]}"
        if len(self._runs) == self._runs.maxlen:      # bound memory on a long shift
            self._forget(self._runs[0])
        self._runs.append(run_id)
        init: TriageState = {
            "run_id": run_id, "encounter": encounter,
            "recheck_vitals": kw.get("recheck_vitals"),
            "baseline_vitals": kw.get("baseline_vitals"),
            "nurse_confirmed_codes": list(kw.get("nurse_confirmed_codes") or []),
            "nurse_dismissed_codes": list(kw.get("nurse_dismissed_codes") or []),
            "prior_candidates": kw.get("prior_candidates"),
            "prior_extraction": kw.get("prior_extraction"),
            "model": self.model, "trace": [],
        }
        return run_id, {"configurable": {"thread_id": run_id}}, init

    def _forget(self, run_id: str) -> None:
        try:
            self.graph.checkpointer.delete_thread(run_id)
        except Exception:
            pass

    async def start(self, encounter: dict[str, Any], **kw: Any) -> dict[str, Any]:
        """Run to the approval gate (or to 'needs_correction'). Returns the public state,
        including ``run_id`` (pass it to ``approve``) and ``awaiting_approval``."""
        run_id, config, init = self._new_run(encounter, **kw)
        await self.graph.ainvoke(init, config)
        snap = await self.graph.aget_state(config)
        return {**_public_state(snap.values), "run_id": run_id,
                "awaiting_approval": "approval" in (snap.next or ())}

    async def astream_start(self, encounter: dict[str, Any], **kw: Any):
        """Async generator yielding live events as each node completes, for the UI:

            {"type": "run", "run_id"}                      once, first
            {"type": "trace", "entry"}                     per tool call
            {"type": "validation", "validation", "blocked"}
            {"type": "proposal", "state"}                  as soon as the score is checked —
                                                           BEFORE any model call
            {"type": "candidates", "candidates", "extraction"}
            {"type": "gate", "state", "awaiting_approval"} last
        """
        run_id, config, init = self._new_run(encounter, **kw)
        yield {"type": "run", "run_id": run_id}
        acc: dict[str, Any] = {}
        async for chunk in self.graph.astream(init, config, stream_mode="updates"):
            if "__interrupt__" in chunk:
                continue
            for node, update in chunk.items():
                if not isinstance(update, dict):
                    continue
                for entry in update.get("trace", []) or []:
                    yield {"type": "trace", "entry": entry}
                acc.update({k: v for k, v in update.items() if k != "trace"})
                if node == "validate":
                    yield {"type": "validation", "validation": update.get("validation"),
                           "blocked": update.get("status") == "needs_correction"}
                elif node == "self_check":
                    yield {"type": "proposal", "state": _public_state(acc), "run_id": run_id}
                elif node == "extract":
                    yield {"type": "candidates", "candidates": update.get("candidates", []),
                           "extraction": update.get("extraction", {})}
        snap = await self.graph.aget_state(config)
        yield {"type": "gate", "run_id": run_id, "state": _public_state(snap.values),
               "awaiting_approval": "approval" in (snap.next or ())}

    async def approve(self, run_id: str, decision: dict[str, Any]) -> dict[str, Any]:
        """Resume the paused run with the clinician's decision, run finalize → END.

        Raises ``ApprovalError`` (with a message for the clinician) if the gate rules are not
        met. These are enforced here so no client can bypass them."""
        config = {"configurable": {"thread_id": run_id}}
        snap = await self.graph.aget_state(config)
        values = snap.values or {}
        if "approval" not in (snap.next or ()):
            raise ApprovalError(
                "This assessment is not waiting for approval — it was already confirmed, "
                "could not be scored, or has been replaced. Assess the patient again.")
        kind = decision.get("decision", "approve")
        name = (decision.get("clinician_name") or "").strip()
        result = values.get("result") or {}
        if kind not in ("approve", "override"):
            raise ApprovalError("Decision must be 'approve' or 'override'.")
        if not name:
            raise ApprovalError("Enter the name of the approving clinician.")
        if kind == "override":
            if decision.get("override_priority") not in _VALID_PRIORITIES:
                raise ApprovalError("Choose the priority to override to.")
            if not (decision.get("reason") or "").strip():
                raise ApprovalError("An override needs a written reason.")
        else:
            if not result.get("complete") or not result.get("proposed_priority"):
                raise ApprovalError(
                    "There is no priority to approve yet — complete the vitals and assess "
                    "again. (A senior clinician can override with a priority and a reason.)")
            unreviewed = [c["label"] for c in values.get("candidates") or []
                          if c.get("status") == "unreviewed"]
            if unreviewed:
                raise ApprovalError(
                    "Review the suggested sign(s) first — mark each as present or not "
                    "present: " + "; ".join(unreviewed) + ".")
            # SATS step 2 is "look for very urgent / urgent signs" for EVERY patient. When
            # the language model has not read the complaint (off, slow, or no complaint), the
            # keyword matcher alone misses most lay phrasings — so the nurse must state that
            # they have checked the sign list themselves.
            model_read_it = (values.get("extraction") or {}).get("mode") == "model+keywords"
            if not model_read_it and not decision.get("signs_checked"):
                raise ApprovalError(
                    "The language model did not read this complaint, so signs may have been "
                    "missed. Check the full sign list, then tick 'I have checked the sign "
                    "list' to confirm.")
            ver = values.get("verification") or {}
            if not (ver.get("all_cited") and ver.get("grounded_in_protocol")):
                raise ApprovalError(
                    "The self-check failed (the evidence does not match the protocol text). "
                    "Do not rely on this proposal — a senior clinician must override.")
        state = await self.graph.ainvoke(Command(resume={**decision, "clinician_name": name}),
                                         config)
        return {**_public_state(state), "approval": state.get("approval"), "run_id": run_id,
                "encounter": state.get("encounter")}

    async def narrate(self, run_id: str, timeout: float = 120.0) -> str | None:
        """Optional model-written handover note for a scored run. Never on the gate path,
        and off by default in the app (``settings.handover_note``).

        The model is given only the priority and the reasons — not the "do now" actions, which
        it has been seen to restate as things already done. The note is then checked before it
        is shown: it must state the proposed priority, name no other priority colour, use no
        diagnostic language, and make no claim that anything was done, given or ordered —
        otherwise it is discarded and the deterministic summary stands alone."""
        if not self._llm:
            return None
        snap = await self.graph.aget_state({"configurable": {"thread_id": run_id}})
        values = snap.values or {}
        result, summary = values.get("result") or {}, values.get("summary")
        priority = result.get("proposed_priority")
        if not summary or not result.get("complete") or not priority:
            return None
        from langchain_core.messages import HumanMessage, SystemMessage
        sys = SystemMessage(content=(
            "You are a triage assistant. Write a handover note of at most two short sentences "
            "for the receiving nurse. Use ONLY the facts given. Do NOT diagnose, do NOT suggest "
            "treatment or tests, do NOT say anything has been done, and do NOT change the "
            "priority."))
        facts = format_text({**summary, "actions": [], "cautions": []})
        msg = HumanMessage(content="Facts:\n" + facts)
        try:
            resp = await asyncio.wait_for(self._llm.ainvoke([sys, msg]), timeout)
        except Exception:
            return None
        text = " ".join((resp.content or "").split())
        others = _VALID_PRIORITIES - {priority}
        ok = (text and len(text) <= 400
              and re.search(rf"\b{priority}\b", text, re.IGNORECASE)
              and not any(re.search(rf"\b{o}\b", text, re.IGNORECASE) for o in others)
              and not re.search(r"diagnos", text, re.IGNORECASE)
              and not _CLAIMS_ACTION.search(text))
        audit.record("narration", encounter_id=values.get("encounter", {}).get("encounter_id"),
                     tool="agent", payload={"run_id": run_id, "accepted": bool(ok), "text": text})
        return text if ok else None

    async def warm_up(self, timeout: float = 300.0) -> bool:
        """Load the model AND prime its prompt cache with the real extraction prompt, ahead of
        the first patient. On CPU the first pass over the sign catalogue can take a minute;
        after this, extraction takes a few seconds. Best-effort; never raises."""
        if not self._llm:
            return False
        try:
            await asyncio.wait_for(self._llm_extract("routine review, feels well"), timeout)
            return True
        except Exception:
            return False


def _public_state(values: dict[str, Any]) -> dict[str, Any]:
    """Trim internal channels to what the UI needs."""
    keys = ("result", "summary", "verification", "explanation", "candidates", "extraction",
            "validation", "deterioration", "status", "trace")
    return {k: values.get(k) for k in keys}
