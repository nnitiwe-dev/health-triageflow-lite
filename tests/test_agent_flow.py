"""End-to-end agent flow: validation → score → self-check → extraction → approval gate.

Uses the in-process tool surface (same tool functions, no subprocess) so the whole flow —
including every gate rule — runs offline in milliseconds. These tests pin the failures an
audit found in the earlier design: stale results, approving an unreviewed or incomplete
assessment, and silent fallbacks.
"""

import asyncio

import pytest

from triageflow.agent.graph import ApprovalError, TriageAgent, verify_against_protocol
from triageflow.agent.mcp_client import InProcessMCP, read_protocol_file
from triageflow.core import sats_rules as R
from triageflow.core.models import Encounter, VitalSigns
from triageflow.core.scoring import score_encounter

H = dict(respiratory_rate=12, heart_rate=72, systolic_bp=122, temperature_c=36.7,
         avpu="alert", mobility="walking", trauma=False)
NURSE = {"decision": "approve", "clinician_name": "Nurse A", "signs_checked": True}


def run(coro):
    return asyncio.run(coro)


def enc(eid="T1", complaint="", age=45, **vit):
    return {"encounter_id": eid, "age_years": age, "presenting_complaint": complaint,
            "vitals": {**H, **vit}}


def agent():
    return TriageAgent(InProcessMCP())


class FakeLLM:
    """Stands in for the chat model: returns a canned reply, optionally after a delay."""

    def __init__(self, reply="none", delay=0.0):
        self.reply, self.delay, self.calls = reply, delay, 0

    async def ainvoke(self, _messages):
        self.calls += 1
        await asyncio.sleep(self.delay)
        return type("Resp", (), {"content": self.reply})()


# ---- the gate -------------------------------------------------------------------------------

def test_suggested_sign_must_be_reviewed_before_approval():
    async def go():
        a = agent()
        st = await a.start(enc(complaint="central chest pain radiating to left arm"))
        assert st["result"]["proposed_priority"] == "GREEN"
        assert [(c["code"], c["status"]) for c in st["candidates"]] == [("chest_pain", "unreviewed")]
        with pytest.raises(ApprovalError, match="Review the suggested sign"):
            await a.approve(st["run_id"], NURSE)
        # confirmed -> ORANGE, and now it can be approved
        st = await a.start(enc(complaint="central chest pain radiating to left arm"),
                           nurse_confirmed_codes=["chest_pain"])
        assert st["result"]["proposed_priority"] == "ORANGE"
        fin = await a.approve(st["run_id"], NURSE)
        assert fin["approval"]["confirmed"]["confirmed_priority"] == "ORANGE"
    run(go())


def test_dismissed_sign_no_longer_blocks_and_does_not_score():
    async def go():
        a = agent()
        st = await a.start(enc(complaint="chest pain"), nurse_dismissed_codes=["chest_pain"])
        assert st["candidates"][0]["status"] == "absent"
        fin = await a.approve(st["run_id"], NURSE)
        assert fin["approval"]["confirmed"]["confirmed_priority"] == "GREEN"
    run(go())


def test_incomplete_assessment_cannot_be_approved_only_overridden_with_a_reason():
    async def go():
        a = agent()
        e = {"encounter_id": "T2", "age_years": 40, "vitals": {"heart_rate": 88}}
        st = await a.start(e)
        assert st["summary"]["status"] == "incomplete"
        with pytest.raises(ApprovalError, match="no priority to approve"):
            await a.approve(st["run_id"], NURSE)
        with pytest.raises(ApprovalError, match="written reason"):
            await a.approve(st["run_id"], {"decision": "override", "clinician_name": "Dr B",
                                           "override_priority": "YELLOW", "reason": "  "})
        with pytest.raises(ApprovalError, match="priority to override to"):
            await a.approve(st["run_id"], {"decision": "override", "clinician_name": "Dr B",
                                           "override_priority": "PURPLE", "reason": "x"})
        fin = await a.approve(st["run_id"], {"decision": "override", "clinician_name": "Dr B",
                                             "override_priority": "YELLOW",
                                             "reason": "looks unwell, vitals pending"})
        c = fin["approval"]["confirmed"]
        assert c["confirmed_priority"] == "YELLOW" and c["decision"] == "override"
    run(go())


def test_gate_needs_a_named_clinician_and_fires_once():
    async def go():
        a = agent()
        st = await a.start(enc())
        with pytest.raises(ApprovalError, match="name"):
            await a.approve(st["run_id"], {**NURSE, "clinician_name": " "})
        await a.approve(st["run_id"], NURSE)
        with pytest.raises(ApprovalError, match="not waiting for approval"):
            await a.approve(st["run_id"], NURSE)          # second click: refused, not repeated
        with pytest.raises(ApprovalError):
            await a.approve("NEVER-SEEN::0000", NURSE)   # unknown run: refused, not a crash
    run(go())


# ---- no stale results -----------------------------------------------------------------------

def test_a_failed_validation_never_shows_the_previous_result():
    async def go():
        a = agent()
        first = await a.start(enc("OPD-1", "BP check"))
        assert first["result"]["proposed_priority"] == "GREEN"
        second = await a.start(enc("OPD-1", "now very sick", heart_rate=1300, respiratory_rate=34))
        assert second["status"] == "needs_correction"
        assert second["result"] is None and second["summary"] is None
        assert second["awaiting_approval"] is False
        assert "not possible" in second["validation"]["issues"][0]["message"]
        assert first["run_id"] != second["run_id"]
        with pytest.raises(ApprovalError):
            await a.approve(second["run_id"], NURSE)
    run(go())


def test_emergency_sign_is_not_held_up_by_a_mistyped_vital():
    async def go():
        st = await agent().start({"encounter_id": "T3", "age_years": 60,
                                  "vitals": {"heart_rate": 1300}},
                                 nurse_confirmed_codes=["cardiac_arrest"])
        assert st["status"] == "ok" and st["result"]["proposed_priority"] == "RED"
        assert any("not possible" in c for c in st["summary"]["cautions"])
    run(go())


def test_child_is_blocked_with_the_reason():
    async def go():
        st = await agent().start(enc(age=4))
        assert st["status"] == "needs_correction"
        assert "paediatric" in st["validation"]["issues"][0]["message"].lower()
    run(go())


def test_schema_invalid_value_becomes_a_readable_issue_not_a_crash():
    async def go():
        st = await agent().start(enc(pain_score=14))
        assert st["status"] == "needs_correction"
        assert "Pain score" in st["validation"]["issues"][0]["message"]
    run(go())


# ---- streaming order: the priority is on screen before any model call ----------------------

def test_proposal_is_emitted_before_extraction():
    async def go():
        a = agent()
        a._llm = FakeLLM("chest_pain")
        kinds = [ev["type"] async for ev in a.astream_start(enc(complaint="heart attack?"))]
        assert kinds[0] == "run" and kinds[-1] == "gate"
        assert kinds.index("proposal") < kinds.index("candidates")
    run(go())


# ---- extraction: model, fallback, reuse -----------------------------------------------------

def test_model_suggestions_are_filtered_to_real_codes_and_marked():
    async def go():
        a = agent()
        a._llm = FakeLLM("Codes: chest_pain, made_up_code, SOB_ACUTE.")
        st = await a.start(enc(complaint="feels like an elephant on my chest"))
        got = {c["code"]: c["via"] for c in st["candidates"]}
        assert got == {"chest_pain": "model", "sob_acute": "model"}
        assert st["extraction"]["mode"] == "model+keywords"
    run(go())


def test_slow_model_falls_back_to_keywords_and_says_so():
    async def go():
        a = TriageAgent(InProcessMCP(), extract_timeout=0.05)
        a._llm = FakeLLM("chest_pain", delay=1.0)
        st = await a.start(enc(complaint="central chest pain"))
        assert [c["code"] for c in st["candidates"]] == ["chest_pain"]      # keyword still works
        assert st["extraction"]["mode"] == "keywords"
        assert "took too long" in st["extraction"]["note"]
    run(go())


def test_model_off_is_stated_not_hidden():
    async def go():
        st = await agent().start(enc(complaint="feels like an elephant on my chest"))
        assert st["candidates"] == []
        assert "full sign list" in st["extraction"]["note"]
    run(go())


def test_rescore_reuses_reviewed_candidates_without_calling_the_model():
    async def go():
        a = agent()
        a._llm = FakeLLM("chest_pain")
        first = await a.start(enc(complaint="heart attack?"))
        prior = [{k: v for k, v in c.items() if k != "status"} for c in first["candidates"]]
        calls = a._llm.calls
        second = await a.start(enc(complaint="heart attack?"), prior_candidates=prior,
                               nurse_confirmed_codes=["chest_pain"])
        assert a._llm.calls == calls
        assert second["result"]["proposed_priority"] == "ORANGE"
        assert second["candidates"][0]["status"] == "present"
    run(go())


def test_high_glucose_offers_the_diabetic_signs_for_the_nurse_to_decide():
    async def go():
        st = await agent().start(enc(glucose_mmol=19.2))
        got = {c["code"]: (c["via"], c["status"]) for c in st["candidates"]}
        assert got == {"diabetic_gt11_ketonuria": ("vitals", "unreviewed"),
                       "diabetic_gt17": ("vitals", "unreviewed")}
        assert st["result"]["proposed_priority"] == "GREEN"       # never auto-applied
    run(go())


def test_sign_derived_from_vitals_needs_no_review():
    async def go():
        a = agent()
        st = await a.start(enc(complaint="drowsy since morning", avpu="voice"))
        assert st["result"]["proposed_priority"] == "ORANGE"
        assert [(c["code"], c["status"]) for c in st["candidates"]] == [("loc_reduced", "from_vitals")]
        await a.approve(st["run_id"], NURSE)                      # nothing left to review
    run(go())


# ---- handover note is checked before it is shown --------------------------------------------

@pytest.mark.parametrize("reply,accepted", [
    ("ORANGE, very urgent: chest pain, see within 10 minutes.", True),
    ("This patient is RED and needs resus.", False),              # wrong colour
    ("ORANGE. Probable diagnosis: myocardial infarction.", False),  # diagnostic language
    ("Chest pain, see soon.", False),                             # priority not stated
    ("ORANGE, very urgent. Glucose test done, ECG pending.", False),  # claims actions happened
    ("ORANGE. Oxygen administered.", False),
])
def test_handover_note_check(reply, accepted):
    async def go():
        a = agent()
        st = await a.start(enc(complaint="chest pain"), nurse_confirmed_codes=["chest_pain"])
        a._llm = FakeLLM(reply)
        note = await a.narrate(st["run_id"])
        assert (note == reply) if accepted else (note is None)
    run(go())


# ---- deterioration through the flow ---------------------------------------------------------

def test_recheck_against_a_baseline_raises_the_flag():
    async def go():
        st = await agent().start(enc(respiratory_rate=32, heart_rate=130, avpu="voice"),
                                 baseline_vitals=dict(H))
        det = st["deterioration"]
        assert det["escalated"] is True and det["baseline_tews"] == 0
    run(go())


# ---- self-check: the rule table really does agree with the protocol text --------------------

PROTO = read_protocol_file("sats_adult.md")


def test_protocol_file_reads_as_utf8():
    assert "TEWS" in PROTO and "→" in PROTO


def test_every_discriminator_is_listed_under_its_level_in_the_protocol():
    ev = [{"kind": "discriminator", "parameter": d.code, "band_label": d.label,
           "implies": d.level.value, "source": d.source} for d in R.ALL_DISCRIMINATORS]
    v = verify_against_protocol(ev, PROTO)
    assert v["grounded_in_protocol"], v["notes"]


@pytest.mark.parametrize("vit", [
    dict(respiratory_rate=8, heart_rate=40, systolic_bp=70, temperature_c=34.5, avpu="unresponsive", mobility="immobile", trauma=True),
    dict(respiratory_rate=12, heart_rate=45, systolic_bp=75, temperature_c=36.5, avpu="alert", mobility="walking", trauma=False),
    dict(respiratory_rate=17, heart_rate=105, systolic_bp=90, temperature_c=39.0, avpu="voice", mobility="with_help", trauma=False),
    dict(respiratory_rate=25, heart_rate=120, systolic_bp=150, temperature_c=37.0, avpu="confused", mobility="walking", trauma=False),
    dict(respiratory_rate=35, heart_rate=140, systolic_bp=210, temperature_c=37.0, avpu="pain", mobility="walking", trauma=False),
])
def test_every_tews_band_sits_in_the_right_column_of_the_protocol_table(vit):
    res = score_encounter(Encounter(encounter_id="V", age_years=40, vitals=VitalSigns(**vit)))
    ev = [e.model_dump(mode="json") for e in res.evidence]
    v = verify_against_protocol(ev, PROTO)
    assert v["grounded_in_protocol"] and v["all_cited"], v["notes"]


def test_self_check_catches_a_wrong_point_value_and_blocks_plain_approval():
    res = score_encounter(Encounter(encounter_id="V", age_years=40,
                                    vitals=VitalSigns(**{**H, "respiratory_rate": 25})))
    ev = [e.model_dump(mode="json") for e in res.evidence]
    next(e for e in ev if e["parameter"] == "respiratory_rate")["points"] = 0   # tamper
    assert not verify_against_protocol(ev, PROTO)["grounded_in_protocol"]

    async def go():
        a = agent()
        st = await a.start(enc())

        async def empty(_name="sats_adult.md"):
            return ""
        a.mcp.read_protocol = empty                         # protocol text unavailable
        st = await a.start(enc())
        assert st["verification"]["grounded_in_protocol"] is False
        with pytest.raises(ApprovalError, match="self-check failed"):
            await a.approve(st["run_id"], NURSE)
    run(go())


def test_local_rule_is_reported_as_local_not_as_sats():
    res = score_encounter(Encounter(encounter_id="V", age_years=40,
                                    vitals=VitalSigns(**{**H, "spo2": 88})))
    v = verify_against_protocol([e.model_dump(mode="json") for e in res.evidence], PROTO)
    assert v["grounded_in_protocol"] and len(v["local_rules"]) == 1


# ---- when the model has not read the complaint, the nurse must check the sign list ----------

def test_keyword_only_extraction_needs_the_sign_list_check():
    async def go():
        a = agent()                                             # model off
        st = await a.start(enc(complaint="feels like an elephant sitting on his chest"))
        assert st["candidates"] == [] and st["extraction"]["mode"] == "keywords"
        with pytest.raises(ApprovalError, match="did not read this complaint"):
            await a.approve(st["run_id"], {"decision": "approve", "clinician_name": "Nurse A"})
        fin = await a.approve(st["run_id"], NURSE)
        assert fin["approval"]["confirmed"]["confirmed_priority"] == "GREEN"
    run(go())


def test_model_read_complaint_needs_no_extra_tick_and_reuse_remembers_how_it_was_read():
    async def go():
        a = agent()
        a._llm = FakeLLM("none")
        st = await a.start(enc(complaint="here for BP refill, feels well"))
        assert st["extraction"]["mode"] == "model+keywords"
        again = await a.start(enc(complaint="here for BP refill, feels well"),
                              prior_candidates=[], prior_extraction=st["extraction"])
        assert again["extraction"]["mode"] == "model+keywords" and again["extraction"]["reused"]
        await a.approve(again["run_id"], {"decision": "approve", "clinician_name": "Nurse A"})
    run(go())


def test_override_does_not_need_the_tick():
    async def go():
        a = agent()
        st = await a.start(enc(complaint="looks very unwell"))
        fin = await a.approve(st["run_id"], {"decision": "override", "clinician_name": "Dr B",
                                             "override_priority": "ORANGE", "reason": "looks septic"})
        assert fin["approval"]["confirmed"]["confirmed_priority"] == "ORANGE"
    run(go())


def test_handover_note_prompt_leaves_out_the_actions_it_could_misreport():
    seen = {}

    class Spy(FakeLLM):
        async def ainvoke(self, messages):
            seen["facts"] = messages[-1].content
            return await super().ainvoke(messages)

    async def go():
        a = agent()
        st = await a.start(enc(complaint="chest pain"), nurse_confirmed_codes=["chest_pain"])
        assert any("ECG" in x["text"] for x in st["summary"]["actions"])
        a._llm = Spy("ORANGE, very urgent: chest pain.")
        await a.narrate(st["run_id"])
        assert "Chest pain" in seen["facts"] and "ECG" not in seen["facts"]
    run(go())
