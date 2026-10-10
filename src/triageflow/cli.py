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
    from .agent import ollama_manager as om
    from .core.config import settings
    print(f"Model (configured): {settings.model}  | fallback: {settings.fallback_model}")
    st = om.ensure_ready(pull=False, on_progress=lambda s: print("  …", s))
    print(f"Ollama installed : {st.ollama_installed}")
    print(f"Ollama server up : {st.server_up}")
    print(f"Local models     : {st.available_models}")
    print(f"Resolved model   : {st.model}  (fallback={st.using_fallback})")
    print(f"Status           : {st.message}")

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


def _pull(force_clean: bool) -> int:
    from .agent import ollama_manager as om
    st = om.ensure_ready(pull=True, force_clean=force_clean,
                         on_progress=lambda s: print("  …", s))
    print(st.message)
    return 0 if st.model else 1


def _demo() -> int:
    from .agent import ollama_manager as om
    from .agent.graph import TriageAgent
    from .agent.mcp_client import TriageMCP
    from .data import synthetic

    async def run():
        import os
        st = om.ensure_ready(pull=False)
        mcp = TriageMCP(include_filesystem=True)
        await mcp.connect()
        model = None if os.environ.get("TRIAGEFLOW_NO_LLM") else st.model
        agent = TriageAgent(mcp, model=model)
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
    pm = sub.add_parser("pull-model", help="download the open-weights model")
    pm.add_argument("--force-clean", action="store_true", help="clear a broken download first")
    sub.add_parser("demo", help="run one assessment in the terminal")
    sf = sub.add_parser("seed-fhir", help="push synthetic patients to the configured FHIR server")
    sf.add_argument("-n", type=int, default=3, help="how many patients to seed")
    args = p.parse_args(argv)

    if args.cmd == "doctor":
        return _doctor()
    if args.cmd == "seed-fhir":
        return _seed_fhir(args.n)
    if args.cmd == "pull-model":
        return _pull(args.force_clean)
    if args.cmd == "demo":
        return _demo()
    # default: serve
    return _serve(getattr(args, "host", settings.host), getattr(args, "port", settings.port))


if __name__ == "__main__":
    sys.exit(main())
