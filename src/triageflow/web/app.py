"""FastAPI nurse workspace — structured intake, live agent tool-calls, and the approval gate.

The UI is a single guided triage workspace (not a chat box): intake on the left, the
agent's real tool-call trace streaming in the middle, and the sourced proposal + clinician
approval control on the right. Nothing reaches a queue position without a named approval.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import queue
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..agent import fhir_emr
from ..agent.graph import ApprovalError, TriageAgent
from ..agent.llm_backend import get_backend
from ..agent.mcp_client import InProcessMCP, TriageMCP
from ..core import sats_rules as R
from ..core.audit import get_audit
from ..core.config import settings
from ..core.explain import TARGET_TEXT
from ..core.scoring import score_encounter
from ..data import synthetic

_HERE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(_HERE / "templates"))
audit = get_audit()
# Cache-buster for static assets: changes on every server start, so a restart always serves
# fresh app.js / styles.css (no stale-cached-JS surprises).
_ASSET_V = str(int(time.time()))

@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI):
    await _startup()
    try:
        yield
    finally:
        # Quiet, prompt shutdown (Ctrl+C can arrive while things are still starting):
        # tell the backend first so a model load in progress stops instead of retrying, then
        # cancel the background tasks and WAIT for them so nothing is left half-cancelled.
        backend = _state.get("backend")
        close = getattr(backend, "close", None) or getattr(backend, "_stop", None)
        if close:                     # don't leave llama-server holding RAM after exit
            await asyncio.to_thread(close)
        pending = list(_tasks)
        for task in pending:
            task.cancel()
        if pending:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), 10)


app = FastAPI(title="TriageFlow Lite", lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")

_state: dict[str, Any] = {"agent": None, "mcp": None, "model": None, "backend": None,
                          "using_fallback": False, "servers": [], "confirmed": [],
                          "llm_enabled": False}
_tasks: set[asyncio.Task] = set()   # keep references so background tasks aren't collected


def _sse(obj: dict[str, Any]) -> str:
    return f"data: {json.dumps(obj, default=str)}\n\n"


def _background(coro) -> None:
    task = asyncio.create_task(coro)
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


async def _startup() -> None:
    """Come up FAST and never wait on the network. Only our own (local) MCP server is
    awaited; the borrowed filesystem MCP (npx download) and the model runtime (binary
    provisioning, model load) attach in the background and switch on when they are ready."""
    _state["backend"] = get_backend()
    mcp: Any = TriageMCP(include_filesystem=True)
    try:
        await mcp.connect(wait_for_filesystem=False)
    except Exception:
        # Our own server could not be spawned (very unusual). Keep the desk working with the
        # same tool functions in-process rather than refusing to start.
        mcp = InProcessMCP()
    _state["mcp"] = mcp
    _state["servers"] = list(mcp.connected_servers)
    _state["agent"] = None
    _rebuild_agent(None)
    if isinstance(mcp, TriageMCP):
        _background(_attach_filesystem(mcp))
    _background(_bring_up_model())


async def _attach_filesystem(mcp: TriageMCP) -> None:
    await mcp.connect_filesystem()
    _state["servers"] = list(mcp.connected_servers)


async def _bring_up_model() -> None:
    """Start the local runtime (self-provisioning the llama.cpp binary if needed), pick up a
    ready model, and load it into memory before the first patient. Progress is published to
    the status endpoint so the banner says what is happening instead of offering a download
    for something that is merely still starting."""
    backend = _state["backend"]
    _state["starting"] = True

    def progress(msg: str) -> None:
        _state["startup_msg"] = msg
    try:
        await asyncio.to_thread(lambda: backend.ensure_server(wait_s=0, on_progress=progress))
        await asyncio.to_thread(_maybe_refresh_model)
    except Exception as e:
        _state["startup_msg"] = f"model start-up failed: {type(e).__name__}: {e}"
    finally:
        _state["starting"] = False
    await _warm_up_once()


async def _warm_up_once() -> None:
    """Load the active model and prime its prompt cache — once per model."""
    model = _state.get("model")
    if not _state.get("llm_enabled") or not model or _state.get("warmed") == model \
            or _state.get("warming"):
        return
    _state["warming"] = True
    try:
        await _agent().warm_up()
        _state["warmed"] = model
    finally:
        _state["warming"] = False


def _rebuild_agent(model: str | None) -> None:
    """Point the agent at ``model`` (None = deterministic only). The agent is created once
    and re-pointed in place, so assessments waiting at the approval gate survive a model
    becoming ready or being switched."""
    # TRIAGEFLOW_NO_LLM=1 runs the agent deterministically (fast demos/evals, or LLM-off sites).
    agent_model = None if os.environ.get("TRIAGEFLOW_NO_LLM") else model
    base = _state["backend"].chat_base_url() if _state.get("backend") else None
    if _state.get("agent") is None:
        _state["agent"] = TriageAgent(_state["mcp"], model=agent_model, chat_base_url=base,
                                      extract_timeout=settings.extract_timeout_s)
    else:
        _state["agent"].set_llm(agent_model, base)
    _state["llm_enabled"] = _state["agent"].llm_enabled


def _agent() -> TriageAgent:
    return _state["agent"]


def _maybe_refresh_model() -> None:
    """Self-heal: if no model was ready at startup, re-check — lightweight (no server start,
    no event-loop block). Lets a model that becomes available later clear the banner itself."""
    if os.environ.get("TRIAGEFLOW_NO_LLM") or _state.get("model") or not _state.get("backend"):
        return
    model, using_fallback = _state["backend"].resolve_model()
    if model:
        _state["model"] = model
        _state["using_fallback"] = using_fallback
        _rebuild_agent(model)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    await asyncio.to_thread(_maybe_refresh_model)  # resolve active model before paint (non-blocking)
    return templates.TemplateResponse(request, "index.html", {
        "model": _state["model"] or "detecting…",
        "using_fallback": _state["using_fallback"],
        "servers": _state["servers"],
        "data_region": settings.data_region,
        "asset_v": _ASSET_V,
    })


@app.get("/api/status")
async def status():
    await asyncio.to_thread(_maybe_refresh_model)
    if not _state.get("starting"):       # a model that became ready later (e.g. Ollama) still
        _background(_warm_up_once())     # gets loaded before the first patient needs it
    backend = _state.get("backend")
    bs = await asyncio.to_thread(backend.status) if backend else None
    return {
        "model": _state["model"], "using_fallback": _state["using_fallback"],
        "servers": _state["servers"], "data_region": settings.data_region,
        "model_ready": _state["model"] is not None,
        "llm_enabled": bool(_state.get("llm_enabled")),
        "warming": bool(_state.get("warming")),
        "starting": bool(_state.get("starting")),
        "startup_msg": _state.get("startup_msg", ""),
        "server_up": bs.server_up if bs else False,
        "backend_message": bs.message if bs else "",
        "handover_note": settings.handover_note,
        "llm_off_by_config": bool(os.environ.get("TRIAGEFLOW_NO_LLM")),
        "backend": backend.name if backend else None,
        "runtime_installed": bs.runtime_installed if bs else False,
        "backend_hint": bs.hint if bs else "",
        "catalog": [asdict(o) for o in (bs.catalog if bs else [])],
        "fhir": {"enabled": fhir_emr.fhir_enabled(), "base_url": settings.fhir_base_url or None},
    }


@app.get("/api/models")
async def list_models_endpoint():
    backend = _state["backend"]
    bs = await asyncio.to_thread(backend.status)
    return {"backend": backend.name, "active": _state["model"], "hint": bs.hint,
            "runtime_installed": bs.runtime_installed,
            "catalog": [asdict(o) for o in bs.catalog]}


@app.post("/api/model/select")
async def model_select(request: Request):
    """Switch the active model — downloading it first if needed (streamed progress).
    Works for both backends; for llama.cpp this restarts llama-server with the new GGUF."""
    body = await request.json()
    model_id = (body or {}).get("model")
    backend = _state["backend"]
    if not model_id:
        return JSONResponse({"error": "model is required"}, status_code=400)
    # Runtime readiness (incl. self-provisioning) is handled in the worker so its progress
    # streams to the UI and the user sees a clear install hint if it can't be set up.

    q: queue.Queue[str | None] = queue.Queue()

    def worker():
        try:
            # Make the runtime available (llama.cpp fetches its own binary, streaming progress)
            # WITHOUT loading the previous model first — the chosen one is started below.
            ensure_runtime = getattr(backend, "ensure_runtime", None)
            if ensure_runtime:
                ensure_runtime(on_progress=lambda s: q.put(s))
            if not backend.runtime_installed():
                q.put("__ERR__" + (backend.status().hint or "runtime not installed"))
                q.put(None)
                return
            if not backend.is_installed(model_id):
                q.put(f"preparing {model_id}…")
                if not backend.pull_model(model_id, on_progress=lambda s: q.put(s)):
                    q.put("__ERR__download failed — check your connection and try again")
                    q.put(None)
                    return
            q.put("starting model — loading it into memory can take a minute…")
            if backend.set_active_model(model_id) is False:
                q.put("__ERR__" + (getattr(backend, "last_error", "") or "model failed to start"))
                q.put(None)
                return
            q.put("__DONE__" + json.dumps({"model": backend.active_model()}))
        except Exception as e:
            q.put(f"__ERR__{type(e).__name__}: {e}")
        q.put(None)

    threading.Thread(target=worker, daemon=True).start()

    async def gen():
        while True:
            item = await asyncio.to_thread(q.get)
            if item is None:
                break
            if item.startswith("__DONE__"):
                data = json.loads(item[len("__DONE__"):])
                if data.get("model"):
                    _state["model"] = data["model"]
                    _state["using_fallback"] = False
                    _rebuild_agent(data["model"])
                    _background(_warm_up_once())
                yield _sse({"type": "done", **data})
            elif item.startswith("__ERR__"):
                yield _sse({"type": "error", "message": item[len("__ERR__"):]})
            else:
                yield _sse({"type": "progress", "message": item})

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/fhir/status")
async def fhir_status():
    """Reachability of the configured FHIR server (no writes)."""
    return await asyncio.to_thread(fhir_emr.server_status)


@app.post("/api/fhir/import")
async def fhir_import(request: Request):
    """Import vitals from the FHIR server. With {patient_id} reads that patient; otherwise seeds
    one synthetic patient and reads it back (a full write→read round trip). Returns an encounter
    to populate intake. Degrades gracefully if the server is unavailable."""
    if not fhir_emr.fhir_enabled():
        return JSONResponse({"error": "FHIR integration is off (set TRIAGEFLOW_FHIR_BASE_URL)."},
                            status_code=400)
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    patient_id = (body or {}).get("patient_id")
    try:
        if patient_id:
            vitals = await asyncio.to_thread(fhir_emr.import_patient_vitals, patient_id)
            complaint = ""
        else:
            enc = synthetic.make_encounter("severe", seed=int(__import__("time").time()) % 100000)
            patient_id, vitals = await asyncio.to_thread(fhir_emr.demo_round_trip, enc)
            complaint = enc.presenting_complaint
    except fhir_emr.FHIRUnavailable as e:
        return JSONResponse({"error": f"FHIR server unavailable: {e}"}, status_code=502)
    return {
        "patient_id": patient_id,
        "source": f"FHIR @ {settings.fhir_base_url}",
        "encounter": {
            "encounter_id": f"FHIR-{patient_id}",
            "presenting_complaint": complaint,
            "discriminator_codes": [],
            "vitals": vitals.model_dump(mode="json"),
        },
    }


_ORDER = {"RED": 0, "ORANGE": 1, "YELLOW": 2, "GREEN": 3, "BLUE": 4, None: 5}


@app.get("/api/queue")
async def queue_demo(n: int = 6, seed: int = 42):
    """The waiting queue: every patient confirmed at this desk in this session, plus a
    reproducible synthetic backdrop (clearly marked ``demo``) so the strip is never empty."""
    out = []
    for item in _state["confirmed"]:
        out.append({**item, "demo": False})
    for enc in synthetic.stream(n=n, seed=seed):
        r = score_encounter(enc)
        out.append({
            "demo": True,
            "encounter": enc.model_dump(mode="json"),
            "priority": r.proposed_priority.value if r.proposed_priority else None,
            "tews_total": r.tews_total,
            "complaint": enc.presenting_complaint,
            "arrival": enc.arrival_time.isoformat(),
        })
    # Order by acuity (then arrival) so the most urgent surfaces — the point of the desk.
    out.sort(key=lambda x: (_ORDER.get(x["priority"], 5), x.get("arrival", "")))
    total = len(out)
    for i, item in enumerate(out):
        item["position"] = i + 1
        item["total"] = total
        item["is_next"] = (i == 0)
        p = item["priority"]
        item["acuity"] = R.ACUITY_NAME[R.Priority(p)] if p else ""
        item["target"] = TARGET_TEXT.get(p, "")
    return out


@app.get("/api/synthetic")
async def synthetic_one():
    """One synthetic patient of a RANDOM acuity (so 'Load synthetic' isn't always green).
    Signs are NOT pre-applied: the nurse confirms them, exactly as with a real patient."""
    import random as _r
    # Weighted toward a realistic OPD spread (more routine than critical), but varied.
    profile = _r.choices(["stable", "moderate", "severe", "critical"],
                         weights=[4, 3, 2, 1], k=1)[0]
    enc = synthetic.make_encounter(profile, seed=_r.randint(0, 999999))
    return {"encounter": enc.model_dump(mode="json"), "complaint": enc.presenting_complaint}


@app.get("/api/discriminators")
async def discriminators():
    """The full SATS adult sign list, grouped by level — the checklist the nurse can always
    use, whether or not the extractor suggested anything."""
    def group(level: str, title: str, signs):
        return {"level": level, "title": title,
                "signs": [{"code": d.code, "label": d.label} for d in signs]}
    return [group("RED", "Emergency signs — straight to resus", R.EMERGENCY_SIGNS),
            group("ORANGE", "Very urgent signs", R.VERY_URGENT_SIGNS),
            group("YELLOW", "Urgent signs", R.URGENT_SIGNS)]


@app.get("/api/demo/deterioration")
async def demo_deterioration(seed: int = 7):
    base, recheck = synthetic.deterioration_series(seed)
    return {"encounter": base.model_dump(mode="json"),
            "recheck_vitals": recheck.vitals.model_dump(mode="json")}


@app.post("/api/assess")
async def assess(request: Request):
    """Run the agent to the approval gate, streaming each tool call as it happens (SSE).
    The deterministic proposal is sent as soon as it is scored — before any model call."""
    body = await request.json()
    encounter = dict(body.get("encounter") or {})
    age = encounter.get("age_years")
    if isinstance(age, float) and age == age:      # a number field may send 54.0 or 54.5
        encounter["age_years"] = int(age)

    async def gen():
        try:
            async for ev in _agent().astream_start(
                    encounter,
                    recheck_vitals=body.get("recheck_vitals"),
                    baseline_vitals=body.get("baseline_vitals"),
                    nurse_confirmed_codes=body.get("nurse_confirmed_codes", []),
                    nurse_dismissed_codes=body.get("nurse_dismissed_codes", []),
                    prior_candidates=body.get("prior_candidates"),
                    prior_extraction=body.get("prior_extraction")):
                yield _sse(ev)
        except Exception as e:  # never leave the UI hanging
            yield _sse({"type": "error", "message": f"{type(e).__name__}: {e}"})
        yield _sse({"type": "done"})

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/approve")
async def approve(request: Request):
    """The human gate. The rules are enforced by the agent (server-side), so a refusal comes
    back as a clear message (HTTP 409) rather than a silent or meaningless confirmation."""
    body = await request.json()
    run_id = body.get("run_id")
    if not run_id:
        return JSONResponse({"error": "Assess the patient first."}, status_code=400)
    decision = {
        "decision": body.get("decision", "approve"),
        "clinician_name": (body.get("clinician_name") or "").strip(),
        "override_priority": body.get("override_priority"),
        "reason": (body.get("reason") or "").strip(),
        "signs_checked": bool(body.get("signs_checked")),
    }
    try:
        final = await _agent().approve(run_id, decision)
    except ApprovalError as e:
        return JSONResponse({"error": str(e)}, status_code=409)
    except Exception as e:
        return JSONResponse({"error": f"Could not record the decision: {e}"}, status_code=500)
    confirmed = (final.get("approval") or {}).get("confirmed") or {}
    if not confirmed.get("confirmed_priority"):
        return JSONResponse({"error": "The decision was not recorded — assess again."},
                            status_code=500)
    enc = final.get("encounter") or {}
    result = final.get("result") or {}
    # The irreversible action: this patient now has a place in the queue.
    _state["confirmed"] = [c for c in _state["confirmed"]
                           if c["encounter"].get("encounter_id") != enc.get("encounter_id")]
    _state["confirmed"].append({
        "encounter": enc, "priority": confirmed["confirmed_priority"],
        "proposed": confirmed.get("proposed_priority"), "decision": confirmed.get("decision"),
        "clinician": confirmed.get("clinician"), "tews_total": result.get("tews_total"),
        "complaint": enc.get("presenting_complaint", ""),
        "arrival": enc.get("arrival_time") or time.strftime("%Y-%m-%dT%H:%M:%S"),
        "confirmed_at": time.strftime("%H:%M"),
    })
    return {"status": final.get("status"), "confirmed": confirmed, "trace": final.get("trace", [])}


@app.post("/api/narrate")
async def narrate(request: Request):
    """Optional model-written handover note for a scored run (checked before it is shown).
    Off unless TRIAGEFLOW_HANDOVER_NOTE=1. Returns {"note": null} when disabled, when the
    model is off or slow, or when its note failed the check."""
    body = await request.json()
    if not settings.handover_note:
        return {"note": None}
    try:
        note = await _agent().narrate(body.get("run_id", ""))
    except Exception:
        note = None
    return {"note": note}


@app.get("/api/encounter/{encounter_id}/audit")
async def encounter_audit(encounter_id: str):
    return audit.trail(encounter_id)
