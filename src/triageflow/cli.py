"""TriageFlow Lite command-line entrypoint.

    triageflow serve        # start the nurse workspace (web UI)
    triageflow doctor       # check the model runtime (llama.cpp / Ollama), model, MCP servers
    triageflow pull-model   # download an open-weights model (resumable)
    triageflow demo         # run one assessment in the terminal (tool calls + approval)
"""

from __future__ import annotations

import argparse
import asyncio
import sys


def _utf8_console() -> None:
    """Windows consoles default to a legacy code page that cannot print '…', '✓' or '→' —
    without this the CLI would crash with UnicodeEncodeError on the first status line."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _say(msg: str) -> None:
    print(f"  … {msg}", flush=True)


def _doctor() -> int:
    import platform
    import shutil

    from .agent.llm_backend import get_backend
    from .core.config import settings
    print(f"Platform         : {platform.system()} {platform.machine()} · Python "
          f"{platform.python_version()}")
    print(f"Data directory   : {settings.home}")
    backend = get_backend()
    backend.ensure_server(wait_s=45, on_progress=_say)
    st = backend.status()
    print(f"LLM backend      : {st.backend}  (TRIAGEFLOW_LLM_BACKEND={settings.llm_backend})")
    print(f"Runtime installed: {st.runtime_installed}")
    print(f"Server up        : {st.server_up}")
    print(f"Resolved model   : {st.model}  (fallback={st.using_fallback})")
    print(f"Status           : {st.message}" + (f"  — {st.hint}" if st.hint else ""))
    print("Model catalog    :")
    for m in st.catalog:
        print(f"   [{'✓' if m.installed else ' '}] {m.id:28} {m.label}  (~{m.size_gb} GB)")
    if not st.model:
        print("   (no model running — triage priorities still work; sign suggestions are "
              "keyword-only until a model is downloaded: `triageflow pull-model`)")
    print(f"Node / npx       : {shutil.which('npx') or 'not found — protocol text is read locally instead'}")

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
        # llama.cpp can fetch its own prebuilt runtime; Ollama needs its installer.
        ensure = getattr(backend, "ensure_runtime", None)
        if not (ensure and ensure(on_progress=_say)):
            print(backend.status().hint or "runtime not installed")
            return 1
    target = model_id or (settings.model if backend.name == "ollama"
                          else settings.llamacpp_default_model)
    if backend.is_installed(target):
        print(f"'{target}' is already available to {backend.name} — nothing to download.")
        return 0
    print(f"Downloading '{target}' via {backend.name} …")
    ok = backend.pull_model(target, on_progress=_say)
    print("done." if ok else f"FAILED. {getattr(backend, 'last_error', '')}".rstrip())
    return 0 if ok else 1


def _demo() -> int:
    from .agent.graph import TriageAgent
    from .agent.llm_backend import get_backend
    from .agent.mcp_client import TriageMCP
    from .core.explain import format_text
    from .data import synthetic

    async def run():
        import os
        backend = get_backend()
        backend.ensure_server(wait_s=45, on_progress=_say)
        model_name, _ = backend.resolve_model()
        mcp = TriageMCP(include_filesystem=True)
        await mcp.connect()
        model = None if os.environ.get("TRIAGEFLOW_NO_LLM") else model_name
        agent = TriageAgent(mcp, model=model, chat_base_url=backend.chat_base_url(),
                            extract_timeout=120)
        patient = synthetic.make_scenario("chest_pain", seed=99)
        truth = list(patient.discriminator_codes)      # what the nurse will confirm
        enc = patient.model_dump(mode="json")
        enc.update({"encounter_id": "DEMO-1", "discriminator_codes": []})
        enc["vitals"].pop("measured_at", None)         # taken "now", at the desk
        line = "-" * 78

        def show(state, title):
            print(f"\n{title}\n{line}")
            for i, t in enumerate(state.get("trace", []), 1):
                print(f"  {i}. [{t['step']:<13}] {t['server']:<28} {t['summary']}")
            print(line)
            print(format_text(state["summary"]))

        print(f"\nEncounter DEMO-1 · {enc['age_years']}y {enc['sex']}\n  {enc['presenting_complaint']}")
        first = await agent.start(enc)
        show(first, "1) First pass — vitals only, signs not yet confirmed")
        cands = first.get("candidates") or []
        print("\n  Suggested signs for the nurse to confirm:")
        for c in cands:
            print(f"    ? {c['label']} → {c['implies']}  ({c['via']})")
        if (first.get("extraction") or {}).get("note"):
            print(f"    note: {first['extraction']['note']}")
        confirmed = [c["code"] for c in cands if c["code"] in truth] or truth
        dismissed = [c["code"] for c in cands if c["code"] not in confirmed]
        prior = [{k: v for k, v in c.items() if k != "status"} for c in cands]
        second = await agent.start(enc, nurse_confirmed_codes=confirmed,
                                   nurse_dismissed_codes=dismissed, prior_candidates=prior)
        show(second, f"2) Nurse confirms: {', '.join(confirmed)} — re-scored")
        print("\nPaused at the approval gate — simulating the clinician's approval…")
        final = await agent.approve(second["run_id"],
                                    {"decision": "approve", "clinician_name": "Demo Nurse",
                                     "signs_checked": True})
        c = (final.get("approval", {}) or {}).get("confirmed", {})
        print(f"CONFIRMED: {c.get('confirmed_priority')} by {c.get('clinician')} — "
              "written to the audit trail.")
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
    _utf8_console()

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
