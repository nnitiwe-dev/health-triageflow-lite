"""FastAPI nurse workspace — structured intake, live agent tool-calls, and the approval gate.

The UI is a single guided triage workspace (not a chat box): intake on the left, the
agent's real tool-call trace streaming in the middle, and the sourced proposal + clinician
approval control on the right. Nothing reaches a queue position without a named approval.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..agent import ollama_manager as om
from ..agent.graph import TriageAgent
from ..agent.mcp_client import TriageMCP
from ..core.audit import get_audit
from ..core.config import settings
from ..core.scoring import score_encounter
from ..data import synthetic

_HERE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(_HERE / "templates"))
audit = get_audit()

app = FastAPI(title="TriageFlow Lite")
app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")

_state: dict[str, Any] = {"agent": None, "mcp": None, "model": None,
                          "using_fallback": False, "servers": []}


def _sse(obj: dict[str, Any]) -> str:
    return f"data: {json.dumps(obj, default=str)}\n\n"


@app.on_event("startup")
async def _startup() -> None:
    st = om.ensure_ready(pull=False)
    _state["model"] = st.model
    _state["using_fallback"] = st.using_fallback
    mcp = TriageMCP(include_filesystem=True)
    try:
        await mcp.connect()
        _state["servers"] = mcp.connected_servers
    except Exception:
        _state["servers"] = []
    _state["mcp"] = mcp
    # TRIAGEFLOW_NO_LLM=1 runs the agent deterministically (fast demos/evals, or LLM-off sites).
    agent_model = None if os.environ.get("TRIAGEFLOW_NO_LLM") else st.model
    _state["llm_enabled"] = agent_model is not None
    _state["agent"] = TriageAgent(mcp, model=agent_model)


def _agent() -> TriageAgent:
    return _state["agent"]


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "model": _state["model"] or "none",
        "using_fallback": _state["using_fallback"],
        "servers": _state["servers"],
        "data_region": settings.data_region,
    })


@app.get("/api/status")
async def status():
    return {
        "model": _state["model"], "using_fallback": _state["using_fallback"],
        "servers": _state["servers"], "data_region": settings.data_region,
        "ollama_models": om.list_models(),
        "model_ready": _state["model"] is not None,
    }


@app.get("/api/queue")
async def queue_demo(n: int = 6, seed: int = 42):
    """A reproducible synthetic queue with instant deterministic bands (no LLM)."""
    out = []
    for enc in synthetic.stream(n=n, seed=seed):
        r = score_encounter(enc)
        out.append({
            "encounter": enc.model_dump(mode="json"),
            "proposed_priority": r.proposed_priority.value if r.proposed_priority else None,
            "tews_total": r.tews_total,
            "complaint": enc.presenting_complaint,
        })
    # Order by acuity so the most urgent surfaces — the whole point of the desk.
    order = {"RED": 0, "ORANGE": 1, "YELLOW": 2, "GREEN": 3, None: 4}
    out.sort(key=lambda x: order.get(x["proposed_priority"], 4))
    return out


@app.get("/api/demo/deterioration")
async def demo_deterioration(seed: int = 7):
    base, recheck = synthetic.deterioration_series(seed)
    return {"encounter": base.model_dump(mode="json"),
            "recheck_vitals": recheck.vitals.model_dump(mode="json")}


@app.post("/api/assess")
async def assess(request: Request):
    """Run the agent to the approval gate, streaming each tool call as it happens (SSE)."""
    body = await request.json()
    encounter = body.get("encounter", {})
    recheck = body.get("recheck_vitals")
    confirmed = body.get("nurse_confirmed_codes", [])

    async def gen():
        try:
            async for ev in _agent().astream_start(
                    encounter, recheck_vitals=recheck, nurse_confirmed_codes=confirmed):
                yield _sse(ev)
        except Exception as e:  # never leave the UI hanging
            yield _sse({"type": "error", "message": f"{type(e).__name__}: {e}"})
        yield _sse({"type": "done"})

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/approve")
async def approve(request: Request):
    body = await request.json()
    enc_id = body["encounter_id"]
    decision = {
        "decision": body.get("decision", "approve"),
        "clinician_name": body.get("clinician_name", "").strip(),
        "override_priority": body.get("override_priority"),
        "reason": body.get("reason", ""),
    }
    if not decision["clinician_name"]:
        return JSONResponse({"error": "clinician_name is required"}, status_code=400)
    final = await _agent().approve(enc_id, decision)
    appr = final.get("approval", {})
    return {"status": final.get("status"),
            "confirmed": appr.get("confirmed", {}),
            "trace": final.get("trace", [])}


@app.get("/api/encounter/{encounter_id}/audit")
async def encounter_audit(encounter_id: str):
    return audit.trail(encounter_id)


@app.post("/api/model/pull")
async def model_pull(request: Request):
    """Download the model, streaming progress. Self-heals: clears partial downloads and
    retries. Pass {"force_clean": true} to wipe a broken download and start fresh."""
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    force_clean = bool(body.get("force_clean"))

    q: queue.Queue[str | None] = queue.Queue()

    def worker():
        st = om.ensure_ready(pull=True, force_clean=force_clean,
                             on_progress=lambda s: q.put(s))
        q.put(f"__DONE__{json.dumps({'model': st.model, 'message': st.message})}")
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
                    _state["agent"] = TriageAgent(_state["mcp"], model=data["model"])
                yield _sse({"type": "done", **data})
            else:
                yield _sse({"type": "progress", "message": item})

    return StreamingResponse(gen(), media_type="text/event-stream")
