"""Eval harness — runs the cases, checks expectations, measures run-to-run variance.

    python evals/run.py            # deterministic engine (fast, no model)

Prints a pass/fail table and a determinism check (each case scored 25x must be identical).
Exits non-zero if any case regresses against its expectation (E8 is expected to FAIL).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from evals.cases import CASES  # noqa: E402
from triageflow.core.models import VitalSigns  # noqa: E402
from triageflow.core.scoring import score_encounter  # noqa: E402
from triageflow.core.validation import Severity, validate_vitals  # noqa: E402


def check(case):
    cid, desc, enc, exp, recheck = case
    val = validate_vitals(enc.vitals)
    has_err = any(i.severity == Severity.ERROR for i in val.issues)

    if exp.get("validation_error"):
        ok = has_err
        got = f"validation_error={has_err}"
        return cid, desc, ok, got

    res = score_encounter(enc)
    got_priority = res.proposed_priority.value if res.proposed_priority else None
    checks = []

    if "priority" in exp:
        checks.append(got_priority == exp["priority"])
    if "complete" in exp:
        checks.append(res.complete == exp["complete"])
    if "min_citations" in exp:
        checks.append(res.citation_count() >= exp["min_citations"])
    if "missing_contains" in exp:
        checks.append(exp["missing_contains"] in res.missing)
    if exp.get("deterioration_escalates"):
        from triageflow.mcp.server import check_deterioration
        d = check_deterioration(enc.encounter_id, enc.vitals.model_dump(mode="json"),
                                VitalSigns(**recheck).model_dump(mode="json"))
        checks.append(bool(d["escalated"]))

    ok = all(checks)
    got = f"priority={got_priority} complete={res.complete} cites={res.citation_count()}"
    if res.missing:
        got += f" missing={res.missing}"
    return cid, desc, ok, got


def determinism(case) -> bool:
    _, _, enc, exp, _ = case
    if exp.get("validation_error"):
        return True
    runs = {score_encounter(enc).model_dump_json(exclude={"computed_at"}) for _ in range(25)}
    return len(runs) == 1


def main() -> int:
    print(f"{'ID':4} {'RESULT':8} {'DETERMINISTIC':14} DESCRIPTION")
    print("-" * 92)
    regressions = 0
    known_fail = {"E8"}
    for case in CASES:
        cid, desc, ok, got = check(case)
        det = determinism(case)
        expected_fail = cid in known_fail
        # A known-fail case "passes the suite" when it still fails as documented.
        suite_ok = (not ok) if expected_fail else ok
        label = ("xfail" if expected_fail and not ok else
                 "UNEXPECTED-PASS" if expected_fail and ok else
                 "pass" if ok else "FAIL")
        if not suite_ok:
            regressions += 1
        print(f"{cid:4} {label:8} {'yes' if det else 'NO-VARIES':14} {desc}")
        print(f"     └─ {got}")
    print("-" * 92)
    total = len(CASES)
    print(f"{total} cases · {total - regressions} as-expected · {regressions} regression(s) · "
          f"E8 is a documented known-failure (xfail)")
    return 1 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
