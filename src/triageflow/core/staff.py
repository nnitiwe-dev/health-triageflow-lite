"""A small roster of SIMULATED clinical staff for the demo sign-in.

This is **not** an authentication system and holds **no real people**. It exists so the demo
can drop the "type your name every time" box: the user picks a staff member once, and that name
is what the approval gate records. A production deployment would replace this with real
authenticated users (see docs/DEPLOY_AWS.md and the productionisation notes) — the approval gate
already requires a named approver, so only the *source* of the name changes.

Names, registration numbers and shifts here are invented for the demo. Registration-number
formats loosely follow Kenyan bodies (NCK — Nursing Council of Kenya; COC — Clinical Officers
Council; KMPDC — Medical Practitioners & Dentists Council) to make the demo feel real, but every
entry is fictional.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Staff:
    id: str
    name: str
    role: str          # short role label shown in the UI
    reg_no: str        # (simulated) professional registration number
    shift: str         # Day / Night / On call
    senior: bool = False   # may override a priority at the gate (clinical officer / medical officer)

    def public(self) -> dict:
        return asdict(self)


# A plausible mixed triage team for an out-patient / casualty department.
_ROSTER: list[Staff] = [
    Staff("nurse-achieng", "Grace Achieng'", "Registered Nurse · Triage lead",
          "NCK/RN/2016/0412", "Day"),
    Staff("nurse-otieno", "Samuel Otieno", "Registered Nurse",
          "NCK/RN/2019/0881", "Day"),
    Staff("nurse-hassan", "Amina Hassan", "Registered Nurse · Casualty",
          "NCK/RN/2020/1174", "Night"),
    Staff("nurse-njeri", "Mary Njeri", "Registered Nurse",
          "NCK/RN/2018/0659", "Night"),
    Staff("co-kamau", "David Kamau", "Clinical Officer (senior)",
          "COC/2015/0337", "Day", senior=True),
    Staff("mo-mwangi", "Dr. Faith Mwangi", "Medical Officer (senior)",
          "KMPDC/MO/2014/0098", "On call", senior=True),
]

_BY_ID = {s.id: s for s in _ROSTER}


def roster() -> list[Staff]:
    """The full demo staff list (order is the display order)."""
    return list(_ROSTER)


def get(staff_id: str | None) -> Staff | None:
    return _BY_ID.get(staff_id or "")
