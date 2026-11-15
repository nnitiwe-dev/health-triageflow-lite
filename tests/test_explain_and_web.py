"""Plain-language summary, the approval tool's server-side rules, and the web API."""

import pytest
from fastapi.testclient import TestClient

from triageflow.agent.graph import TriageAgent
from triageflow.agent.mcp_client import InProcessMCP
from triageflow.core.explain import build_summary, format_text
from triageflow.core.models import Encounter, VitalSigns
from triageflow.core.scoring import score_encounter
from triageflow.mcp.server import request_clinician_approval, validate_triage_input
from triageflow.web import app as webapp

H = dict(respiratory_rate=12, heart_rate=72, systolic_bp=122, temperature_c=36.7,
         avpu="alert", mobility="walking", trauma=False)


def summary(codes=(), age=50, **vit):
    enc = Encounter(encounter_id="S", age_years=age, vitals=VitalSigns(**{**H, **vit}),
                    discriminator_codes=list(codes))
    return build_summary(score_encounter(enc).model_dump(mode="json"))


# ---- summary --------------------------------------------------------------------------------

def test_summary_leads_with_priority_time_and_place():
    s = summary(["chest_pain"])
    assert s["headline"] == "ORANGE — Very urgent"
    assert s["target"] == "See within 10 minutes" and s["where"]
    assert s["decided_by"] == ["Chest pain"]
    assert s["sentence"].startswith("ORANGE (very urgent): see within 10 minutes.")


def test_summary_uses_words_a_nurse_uses_not_field_names():
    s = summary(respiratory_rate=28, heart_rate=118, temperature_c=39.2)
    text = format_text(s)
    assert "Fast breathing — 28 /min" in text and "Fast pulse — 118 bpm" in text
    assert "High temperature (fever) — 39.2 °C" in text
    for raw in ("respiratory_rate", "heart_rate", "temperature_c", "28.0"):
        assert raw not in text


def test_summary_marks_what_decided_the_priority():
    s = summary(["chest_pain"], respiratory_rate=22)
    deciding = [r["text"] for r in s["reasons"] if r["deciding"]]
    assert deciding == ["Chest pain"]            # the TEWS +2 contributes but does not decide
    assert s["tews"]["total"] == 2 and s["tews"]["band"] == "GREEN"


def test_summary_tews_only_priority_names_the_tews():
    s = summary(respiratory_rate=32, heart_rate=132, systolic_bp=88, temperature_c=39.6,
                mobility="immobile")
    assert s["priority"] == "RED" and "TEWS 11" in s["decided_by"]


def test_summary_green_reads_as_reassurance_not_jargon():
    assert summary()["sentence"] == (
        "GREEN (routine): see within 4 hours. "
        "No urgent signs and vitals in the routine range (TEWS 0).")


def test_summary_incomplete_says_what_is_still_needed():
    enc = Encounter(encounter_id="S", age_years=40, vitals=VitalSigns(heart_rate=88, trauma=False))
    s = build_summary(score_encounter(enc).model_dump(mode="json"))
    assert s["status"] == "incomplete" and s["priority"] is None
    assert s["sentence"].startswith("No priority can be given yet. Still needed: breathing rate,")
    assert "systolic BP" in s["sentence"] and "alertness (AVPU)" in s["sentence"]


def test_summary_child_is_out_of_scope():
    s = summary(age=5)
    assert s["status"] == "out_of_scope" and "paediatric" in s["sentence"].lower()


def test_summary_carries_actions_and_labels_local_rules():
    s = summary(["chest_pain"], spo2=88)
    assert any("ECG" in a["text"] for a in s["actions"])
    local = next(r for r in s["reasons"] if r["kind"] == "safety")
    assert local["detail"] == "local safety rule, not SATS"


# ---- approval tool: rules enforced in the tool itself --------------------------------------

def test_approval_tool_refuses_what_the_ui_used_to_let_through():
    with pytest.raises(ValueError, match="incomplete"):
        request_clinician_approval("E", None, "Nurse A", "approve")
    with pytest.raises(ValueError, match="reason"):
        request_clinician_approval("E", "RED", "Dr B", "override", "GREEN", "")
    with pytest.raises(ValueError, match="valid override_priority"):
        request_clinician_approval("E", "RED", "Dr B", "override", "MAUVE", "because")
    with pytest.raises(ValueError, match="named approver"):
        request_clinician_approval("E", "RED", "  ", "approve")
    ok = request_clinician_approval("E", "RED", "Dr B", "override", "ORANGE", "stable on review")
    assert ok["confirmed_priority"] == "ORANGE" and ok["reason"] == "stable on review"


def test_validate_tool_turns_schema_errors_into_issues():
    out = validate_triage_input({**H, "pain_score": 14, "avpu": "sleepy"})
    assert out["ok"] is False and len(out["issues"]) == 2
    assert all(i["severity"] == "error" for i in out["issues"])


# ---- web API (no startup: in-process agent, no model, no network) ---------------------------

@pytest.fixture
def client():
    webapp._state.update({"mcp": InProcessMCP(), "agent": TriageAgent(InProcessMCP()),
                          "confirmed": [], "llm_enabled": False})
    return TestClient(webapp.app)        # not used as a context manager: startup is skipped


def _assess(client, encounter, **extra):
    import json
    r = client.post("/api/assess", json={"encounter": encounter, **extra})
    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    return events, next(e for e in events if e["type"] == "run")["run_id"]


def test_assess_stream_and_approval_puts_the_patient_in_the_queue(client):
    enc = {"encounter_id": "WEB-1", "age_years": 60, "presenting_complaint": "chest pain",
           "vitals": dict(H)}
    events, run_id = _assess(client, enc, nurse_confirmed_codes=["chest_pain"])
    gate = events[-2]
    assert gate["type"] == "gate" and gate["awaiting_approval"] is True
    assert gate["state"]["summary"]["priority"] == "ORANGE"
    r = client.post("/api/approve", json={"run_id": run_id, "clinician_name": "Sr. Achieng", "signs_checked": True})
    assert r.status_code == 200 and r.json()["confirmed"]["confirmed_priority"] == "ORANGE"
    mine = [q for q in client.get("/api/queue").json() if not q["demo"]]
    assert len(mine) == 1 and mine[0]["priority"] == "ORANGE"
    assert mine[0]["clinician"] == "Sr. Achieng" and mine[0]["acuity"] == "Very urgent"
    # the same run cannot be confirmed twice
    again = client.post("/api/approve", json={"run_id": run_id, "clinician_name": "Sr. Achieng", "signs_checked": True})
    assert again.status_code == 409 and "not waiting for approval" in again.json()["error"]


def test_approve_refusals_are_explained_never_a_fake_confirmation(client):
    enc = {"encounter_id": "WEB-2", "age_years": 60, "presenting_complaint": "chest pain",
           "vitals": dict(H)}
    _, run_id = _assess(client, enc)                              # sign not reviewed
    r = client.post("/api/approve", json={"run_id": run_id, "clinician_name": "Nurse A"})
    assert r.status_code == 409 and "Review the suggested sign" in r.json()["error"]
    _, run_id = _assess(client, {"encounter_id": "WEB-3", "vitals": {"heart_rate": 80}})
    r = client.post("/api/approve", json={"run_id": run_id, "clinician_name": "Nurse A"})
    assert r.status_code == 409 and "no priority to approve" in r.json()["error"]
    r = client.post("/api/approve", json={"clinician_name": "Nurse A"})
    assert r.status_code == 400
    assert [q for q in client.get("/api/queue").json() if not q["demo"]] == []


def test_validation_failure_stream_carries_the_reason(client):
    events, _ = _assess(client, {"encounter_id": "WEB-4", "age_years": 30,
                                 "vitals": {**H, "temperature_c": 101.2}})
    val = next(e for e in events if e["type"] == "validation")
    assert val["blocked"] is True
    assert "Fahrenheit" in val["validation"]["issues"][0]["message"]
    assert not any(e["type"] == "proposal" for e in events)
    assert events[-2]["awaiting_approval"] is False


def test_discriminator_checklist_is_the_full_sats_list(client):
    from triageflow.core.sats_rules import ALL_DISCRIMINATORS
    groups = client.get("/api/discriminators").json()
    assert [g["level"] for g in groups] == ["RED", "ORANGE", "YELLOW"]
    assert sum(len(g["signs"]) for g in groups) == len(ALL_DISCRIMINATORS)


def test_synthetic_patient_and_queue_endpoints(client):
    one = client.get("/api/synthetic").json()
    assert one["encounter"]["presenting_complaint"]
    q = client.get("/api/queue").json()
    assert [x["position"] for x in q] == list(range(1, len(q) + 1))
    order = {"RED": 0, "ORANGE": 1, "YELLOW": 2, "GREEN": 3}
    assert [order[x["priority"]] for x in q] == sorted(order[x["priority"]] for x in q)


def test_narrate_without_a_model_returns_no_note(client):
    assert client.post("/api/narrate", json={"run_id": "x"}).json() == {"note": None}


def test_fractional_age_from_the_number_field_is_accepted(client):
    events, _ = _assess(client, {"encounter_id": "WEB-5", "age_years": 54.0, "vitals": dict(H)})
    assert not any(e["type"] == "error" for e in events)
    assert events[-2]["state"]["summary"]["priority"] == "GREEN"


def test_agent_is_repointed_in_place_when_the_model_changes(client):
    before = webapp._state["agent"]
    webapp._state["backend"] = type("B", (), {"chat_base_url": lambda self: "http://127.0.0.1:9/v1"})()
    webapp._rebuild_agent("some-model")
    assert webapp._state["agent"] is before and webapp._state["llm_enabled"] is True
    webapp._rebuild_agent(None)
    assert webapp._state["agent"] is before and webapp._state["llm_enabled"] is False
