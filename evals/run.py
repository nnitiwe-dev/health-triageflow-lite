"""Eval harness — scoring, the full agent flow, the synthetic set, and sign-finding recall.

    python evals/run.py            # everything that needs no model (fast; what CI runs)
    python evals/run.py --llm      # also measure sign-finding with the local model

Exits non-zero if any case regresses. X1 is a documented known failure (expected to fail).
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))
# Evals must never write into a real audit log / data directory.
os.environ.setdefault("TRIAGEFLOW_HOME", tempfile.mkdtemp(prefix="triageflow-eval-"))

from evals.cases import BENIGN, FLOW, KNOWN_FAILURE, PARAPHRASES, SCORING  # noqa: E402
from triageflow.agent.graph import ApprovalError, TriageAgent  # noqa: E402
from triageflow.agent.mcp_client import InProcessMCP  # noqa: E402
from triageflow.core.lexicon import suggest_codes  # noqa: E402
from triageflow.core.models import VitalSigns  # noqa: E402
from triageflow.core.scoring import score_encounter  # noqa: E402
from triageflow.core.validation import validate_vitals  # noqa: E402
from triageflow.data.synthetic import SCENARIOS, make_scenario  # noqa: E402

for _stream in (sys.stdout, sys.stderr):      # '→' and '✓' on a Windows console
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

LINE = "-" * 100


def check_scoring(case):
    cid, desc, enc, exp, recheck = case
    val = validate_vitals(enc.vitals, age_years=enc.age_years)
    res = score_encounter(enc)
    got_priority = res.proposed_priority.value if res.proposed_priority else None
    checks = []
    if "validation_error" in exp:
        checks.append((not val.ok) == exp["validation_error"])
    if exp.get("validation_warning"):
        checks.append(val.ok and bool(val.warnings()))
    if "priority" in exp:
        checks.append(got_priority == exp["priority"])
    if "complete" in exp:
        checks.append(res.complete == exp["complete"])
    if "min_citations" in exp:
        checks.append(res.citation_count() >= exp["min_citations"])
    if "missing_contains" in exp:
        checks.append(exp["missing_contains"] in res.missing)
    if "tews" in exp:
        checks.append(res.tews_total == exp["tews"])
    if "action" in exp:
        checks.append(exp["action"] in [a.code for a in res.actions])
    if exp.get("local_rule"):
        checks.append(any(e.kind == "safety_rule" and "NOT a SATS" in e.source
                          for e in res.evidence))
    if exp.get("out_of_scope"):
        checks.append(res.out_of_scope)
    if exp.get("deterioration_escalates"):
        from triageflow.mcp.server import check_deterioration
        d = check_deterioration(enc.encounter_id, enc.vitals.model_dump(mode="json"),
                                VitalSigns(**recheck).model_dump(mode="json"))
        checks.append(bool(d["escalated"]))
    got = f"priority={got_priority} complete={res.complete} TEWS={res.tews_total}"
    if not val.ok:
        got += " validation=ERROR"
    return cid, desc, all(checks), got


def deterministic(case) -> bool:
    enc = case[2]
    runs = {score_encounter(enc).model_dump_json(exclude={"computed_at"}) for _ in range(25)}
    return len(runs) == 1


async def check_flow(agent, case):
    cid, desc, enc, kw, decision, (kind, detail) = case
    st = await agent.start(enc, **kw)
    try:
        fin = await agent.approve(st["run_id"], decision)
        got = "confirmed " + str(fin["approval"]["confirmed"]["confirmed_priority"])
        ok = kind == "confirmed" and fin["approval"]["confirmed"]["confirmed_priority"] == detail
    except ApprovalError as e:
        got = "refused: " + str(e)[:70]
        ok = kind == "refused" and detail.lower() in str(e).lower()
    return cid, desc, ok, got


def synthetic_suite(seeds: int = 50):
    wrong = []
    for s in SCENARIOS:
        for seed in range(seeds):
            r = score_encounter(make_scenario(s.key, seed))
            got = r.proposed_priority.value if r.proposed_priority else None
            if got != s.expected:
                wrong.append((s.key, seed, got))
    return len(SCENARIOS) * seeds, wrong


def recall(find) -> tuple[int, list[str]]:
    misses = [f"{want}: {text}" for text, want in PARAPHRASES if want not in find(text)]
    return len(PARAPHRASES) - len(misses), misses


async def llm_suite() -> int:
    from triageflow.agent.llm_backend import get_backend
    backend = get_backend()
    print(f"\nSIGN-FINDING WITH THE MODEL ({backend.name})\n{LINE}")
    backend.ensure_server(wait_s=60, on_progress=lambda s: print("  …", s, flush=True))
    model, _ = backend.resolve_model()
    if not model:
        print("  no model is running — skipped (download one with `triageflow pull-model`)")
        return 0
    agent = TriageAgent(InProcessMCP(), model=model, chat_base_url=backend.chat_base_url())
    await agent.warm_up()
    hits, both, lat = 0, 0, []
    for text, want in PARAPHRASES:
        t = time.time()
        got = await agent._llm_extract(text)
        lat.append(time.time() - t)
        hit = want in got
        hits += hit
        both += hit or want in suggest_codes(text)      # what the nurse is actually offered
        print(f"  {'hit ' if hit else 'MISS'} {want:26} {got}  | {text}", flush=True)
    false_pos = 0
    for text in BENIGN:
        got = await agent._llm_extract(text)
        false_pos += bool(got)
        if got:
            print(f"  FALSE+ {got}  | {text}")
    lat.sort()
    print(LINE)
    n = len(PARAPHRASES)
    print(f"  model {model}: recall {hits}/{n} = {hits / n:.0%} · model + keywords (as in the "
          f"app) {both}/{n} = {both / n:.0%} · benign notes with a suggestion "
          f"{false_pos}/{len(BENIGN)} · median {lat[len(lat) // 2]:.1f}s, slowest {lat[-1]:.1f}s "
          f"per complaint")
    stop = getattr(backend, "_stop", None)
    if stop:
        stop()
    return 0


def main() -> int:
    regressions = 0

    print(f"SCORING (deterministic engine)\n{LINE}")
    for case in SCORING:
        cid, desc, ok, got = check_scoring(case)
        det = deterministic(case)
        regressions += (not ok) + (not det)
        print(f"{cid:4} {'pass' if ok else 'FAIL':5} {'' if det else 'NOT-DETERMINISTIC '}{desc}\n"
              f"     └─ {got}")

    print(f"\nFLOW (agent run + approval gate)\n{LINE}")
    agent = TriageAgent(InProcessMCP())

    async def flows():
        return [await check_flow(agent, c) for c in FLOW]
    for cid, desc, ok, got in asyncio.run(flows()):
        regressions += not ok
        print(f"{cid:4} {'pass' if ok else 'FAIL':5} {desc}\n     └─ {got}")

    print(f"\nSYNTHETIC SET (every scenario × 50 seeds scores its intended colour)\n{LINE}")
    n, wrong = synthetic_suite()
    regressions += bool(wrong)
    print(f"     {'pass' if not wrong else 'FAIL':5} {n - len(wrong)}/{n} patients correct"
          + (f" — first mismatches: {wrong[:3]}" if wrong else ""))

    print(f"\nSIGN-FINDING, KEYWORDS ONLY (model off) — held-out lay phrasings\n{LINE}")
    hits, misses = recall(suggest_codes)
    false_pos = [t for t in BENIGN if suggest_codes(t)]
    regressions += bool(false_pos)
    print(f"     recall {hits}/{len(PARAPHRASES)} = {hits / len(PARAPHRASES):.0%}   "
          f"(a measurement, not a pass mark — see the known failure below)")
    print(f"     {'pass' if not false_pos else 'FAIL':5} benign notes with a suggestion: "
          f"{len(false_pos)}/{len(BENIGN)}")

    print(f"\nKNOWN FAILURE (not fixed)\n{LINE}")
    cid, desc, enc, clinically = KNOWN_FAILURE
    st = asyncio.run(agent.start(enc))
    got = st["result"]["proposed_priority"]
    still_fails = got != clinically and not st["candidates"]
    regressions += not still_fails      # if this ever passes, the docs are stale: look at it
    print(f"{cid:4} {'xfail' if still_fails else 'UNEXPECTED-PASS':5} {desc}\n"
          f"     └─ proposed {got}, clinically {clinically}; suggested: "
          f"{[c['code'] for c in st['candidates']]}; note shown: "
          f"{bool(st['extraction']['note'])}; gate needs the sign-list check: yes (F7)")

    if "--llm" in sys.argv:
        asyncio.run(llm_suite())

    total = len(SCORING) + len(FLOW) + 3
    print(f"\n{LINE}\n{total} checks · {regressions} regression(s) · X1 is a documented "
          f"known failure (xfail)")
    return 1 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
