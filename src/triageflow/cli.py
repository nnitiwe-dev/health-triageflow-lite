"""TriageFlow Lite command-line entrypoint.

    triageflow serve        # start the nurse workspace (web UI)
    triageflow doctor       # check Ollama + model + MCP servers
    triageflow pull-model   # download the open-weights model (self-healing)
    triageflow demo         # run one assessment in the terminal (tool calls + approval)
"""

from __future__ import annotations

import argparse
import asyncio
import sys


def _doctor() -> int:
    from .agent.llm_backend import get_backend
    backend = get_backend()
    backend.ensure_server(wait_s=45)
    st = backend.status()
    print(f"LLM backend      : {st.backend}")
    print(f"Runtime installed: {st.runtime_installed}")
    print(f"Server up        : {st.server_up}")
    print(f"Resolved model   : {st.model}  (fallback={st.using_fallback})")
    print(f"Status           : {st.message}" + (f"  — {st.hint}" if st.hint else ""))
    print("Model catalog    :")
    for m in st.catalog:
        print(f"   [{'✓' if m.installed else ' '}] {m.id:28} {m.label}  (~{m.size_gb} GB)")

    async def _mcp():
        from .agent.mcp_client import TriageMCP
        m = TriageMCP(include_filesystem=True)
        await m.connect()
        print(f"MCP servers      : {m.connected_servers}")
        print(f"MCP tools        : {sorted(t for t in m.tools)}")
    try:
        asyncio.run(_mcp())
    except Exception as e:
        print(f"MCP check failed : {type(e).__name__}: {e}")
        return 1

    from .agent import fhir_emr
    if fhir_emr.fhir_enabled():
        s = fhir_emr.server_status()
        print(f"FHIR integration : ON  → {s['base_url']}  (reachable={s['reachable']})")
    else:
        print("FHIR integration : off (set TRIAGEFLOW_FHIR_BASE_URL to enable EMR import)")
    return 0


def _seed_fhir(n: int) -> int:
    from .agent import fhir_emr
    from .data import synthetic
    if not fhir_emr.fhir_enabled():
        print("Set TRIAGEFLOW_FHIR_BASE_URL first (e.g. https://hapi.fhir.org/baseR4).")
        return 1
    print(f"Seeding {n} synthetic patient(s) to FHIR server "
          f"{fhir_emr.server_status()['base_url']} …")
    for i, enc in enumerate(synthetic.stream(n=n, seed=100 + n), 1):
        try:
            pid = fhir_emr.seed_patient(enc)
            print(f"  {i}. Patient/{pid}  ← {enc.presenting_complaint[:50]}")
        except fhir_emr.FHIRUnavailable as e:
            print(f"  {i}. FAILED: {e}")
            return 1
    return 0


def _pull(model_id: str | None) -> int:
    from .agent.llm_backend import get_backend
    from .core.config import settings
    backend = get_backend()
    if not backend.runtime_installed():
        print(backend.status().hint or "runtime not installed")
        return 1
    target = model_id or (settings.model if backend.name == "ollama"
                          else settings.llamacpp_default_model)
    print(f"Downloading '{target}' via {backend.name} …")
    ok = backend.pull_model(target, on_progress=lambda s: print("  …", s))
    print("done." if ok else "FAILED.")
    return 0 if ok else 1


def _demo() -> int:
    from .agent.graph import TriageAgent
    from .agent.llm_backend import get_backend
    from .agent.mcp_client import TriageMCP
    from .data import synthetic

    async def run():
        import os
        backend = get_backend()
        backend.ensure_server(wait_s=45)
        model_name, _ = backend.resolve_model()
        mcp = TriageMCP(include_filesystem=True)
        await mcp.connect()
        model = None if os.environ.get("TRIAGEFLOW_NO_LLM") else model_name
        agent = TriageAgent(mcp, model=model, chat_base_url=backend.chat_base_url())
        enc = synthetic.make_encounter("severe", seed=99).model_dump(mode="json")
        enc["encounter_id"] = "DEMO-1"
        print(f"\nEncounter {enc['encounter_id']}: {enc['presenting_complaint']}")
        print("-" * 64)
        state = await agent.start(enc)
        for i, s in enumerate(state.get("trace", []), 1):
            print(f"  {i}. [{s['step']:<13}] {s['server']:<26} {s['summary']}")
        r = state.get("result", {})
        print("-" * 64)
        print(f"PROPOSED: {r.get('proposed_priority')}  ({r.get('target_time')})  "
              f"TEWS {r.get('tews_total')}  · {r.get('citation_count', len(r.get('evidence', [])))} citations")
        print(f"Explanation: {state.get('explanation')}")
        print("\nPaused at approval gate — simulating clinician approval…")
        final = await agent.approve("DEMO-1", {"decision": "approve", "clinician_name": "Demo Nurse"})
        c = (final.get("approval", {}) or {}).get("confirmed", {})
        print(f"CONFIRMED: {c.get('confirmed_priority')} by {c.get('clinician')} — logged to audit trail.")
    asyncio.run(run())
    return 0


def _serve(host: str, port: int) -> int:
    import uvicorn
    print(f"TriageFlow Lite → http://{host}:{port}")
    uvicorn.run("triageflow.web.app:app", host=host, port=port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    from .core.config import settings
    p = argparse.ArgumentParser(prog="triageflow", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("serve", help="start the web UI")
    s.add_argument("--host", default=settings.host)
    s.add_argument("--port", type=int, default=settings.port)
    sub.add_parser("doctor", help="check the stack")
    pm = sub.add_parser("pull-model", help="download an open-weights model (default: configured one)")
    pm.add_argument("model", nargs="?", default=None, help="model id from the backend catalog")
    sub.add_parser("demo", help="run one assessment in the terminal")
    sf = sub.add_parser("seed-fhir", help="push synthetic patients to the configured FHIR server")
    sf.add_argument("-n", type=int, default=3, help="how many patients to seed")
    args = p.parse_args(argv)

    if args.cmd == "doctor":
        return _doctor()
    if args.cmd == "seed-fhir":
        return _seed_fhir(args.n)
    if args.cmd == "pull-model":
        return _pull(args.model)
    if args.cmd == "demo":
        return _demo()
    # default: serve
    return _serve(getattr(args, "host", settings.host), getattr(args, "port", settings.port))


if __name__ == "__main__":
    sys.exit(main())
