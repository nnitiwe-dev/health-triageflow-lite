"""Input validation for triage observations.

Separates *data quality* problems (missing, impossible, extreme, stale, contradictory,
wrong-unit) from *scoring*. The agent calls this first and surfaces every issue to the nurse.

Two tiers, deliberately:

  * **ERROR — impossible.** No living or dead patient produces this number (HR 1300,
    SpO2 140, temperature 101 "°C"). It is an entry or unit mistake, so scoring is blocked
    until it is corrected.
  * **WARNING — extreme but possible.** RR 3, SBP 35, temperature 27.5. These are exactly
    the sickest patients; rejecting them as "typos" would hide an emergency. They are scored,
    and the nurse is told to confirm the reading.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel

from . import sats_rules as R
from .models import VitalSigns
from .scoring import REQUIRED_TEWS_PARAMS, fmt_num

# field -> (label, unit, impossible_low, impossible_high, extreme_low, extreme_high)
#   value < impossible_low or > impossible_high  => ERROR (cannot be a real reading)
#   value < extreme_low    or > extreme_high     => WARNING (confirm; life-threatening if real)
LIMITS: dict[str, tuple[str, str, float, float, float, float]] = {
    "respiratory_rate": ("Breathing rate", "/min", 0, 80, 6, 45),
    "heart_rate": ("Pulse rate", "bpm", 0, 300, 30, 200),
    "systolic_bp": ("Systolic BP", "mmHg", 0, 320, 60, 250),
    "diastolic_bp": ("Diastolic BP", "mmHg", 0, 250, 30, 150),
    "temperature_c": ("Temperature", "°C", 20.0, 45.0, 32.0, 41.0),
    "spo2": ("SpO2", "%", 0, 100, 75, 100),
    "glucose_mmol": ("Glucose", "mmol/L", 0, 80, 1.5, 33.0),
    "weight_kg": ("Weight", "kg", 1, 500, 25, 250),
}
STALE_MINUTES = 30  # vitals older than this at triage are flagged for re-check


class Severity(str, Enum):
    ERROR = "error"      # blocks a defensible score
    WARNING = "warning"  # proceed, but surface to the nurse


class Issue(BaseModel):
    field: str
    severity: Severity
    message: str


class ValidationReport(BaseModel):
    ok: bool
    issues: list[Issue] = []
    missing_required: list[str] = []

    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]


def _range_issue(field: str, val: float) -> Issue | None:
    label, unit, imp_lo, imp_hi, ext_lo, ext_hi = LIMITS[field]
    shown = f"{fmt_num(val)} {unit}".strip()
    if field == "temperature_c" and 45 < val <= 115:
        return Issue(field=field, severity=Severity.ERROR,
                     message=f"Temperature {fmt_num(val)} looks like Fahrenheit — enter it in °C.")
    if val < imp_lo or val > imp_hi:
        return Issue(field=field, severity=Severity.ERROR,
                     message=f"{label} {shown} is not possible — check the entry "
                             f"(accepted range {fmt_num(imp_lo)}–{fmt_num(imp_hi)}).")
    if field == "respiratory_rate" and val == 0:
        return Issue(field=field, severity=Severity.WARNING,
                     message="Breathing rate 0 means not breathing — if correct, tick the "
                             "emergency sign 'Obstructed airway / not breathing' (RED).")
    if field == "heart_rate" and val == 0:
        return Issue(field=field, severity=Severity.WARNING,
                     message="Pulse rate 0 means no pulse — if correct, tick the emergency "
                             "sign 'Cardiac arrest' (RED).")
    if val < ext_lo or val > ext_hi:
        return Issue(field=field, severity=Severity.WARNING,
                     message=f"{label} {shown} is extreme — re-check the reading. "
                             "It has been scored as entered; if correct, escalate now.")
    return None


def validate_vitals(vitals: VitalSigns, now: datetime | None = None,
                    age_years: int | None = None) -> ValidationReport:
    now = now or datetime.now(UTC)
    issues: list[Issue] = []
    missing = [p for p in REQUIRED_TEWS_PARAMS if getattr(vitals, p) is None]

    # Scope: the adult chart is for patients older than 12 years / taller than 150 cm.
    if age_years is not None:
        if age_years < 0 or age_years > 125:
            issues.append(Issue(field="age_years", severity=Severity.ERROR,
                message=f"Age {age_years} is not possible — check the entry."))
        elif age_years < R.ADULT_MIN_AGE_YEARS:
            issues.append(Issue(field="age_years", severity=Severity.ERROR,
                message=f"Patient is {age_years} years old — this tool implements the ADULT "
                        "SATS chart only. Use the paediatric SATS chart for this child."))

    # Impossible / extreme values and unit sanity
    for field in LIMITS:
        val = getattr(vitals, field)
        if val is None:
            continue
        issue = _range_issue(field, val)
        if issue:
            issues.append(issue)

    # Staleness
    if vitals.measured_at is not None:
        age_min = (now - vitals.measured_at).total_seconds() / 60
        if age_min > STALE_MINUTES:
            issues.append(Issue(field="measured_at", severity=Severity.WARNING,
                message=f"Vitals were measured {int(age_min)} min ago "
                        f"(more than {STALE_MINUTES}) — consider re-checking them."))

    # BP sanity: diastolic should be below systolic
    if vitals.diastolic_bp is not None and vitals.systolic_bp is not None \
            and vitals.diastolic_bp >= vitals.systolic_bp:
        issues.append(Issue(field="diastolic_bp", severity=Severity.ERROR,
            message=f"Diastolic BP ({fmt_num(vitals.diastolic_bp)}) is not below systolic "
                    f"({fmt_num(vitals.systolic_bp)}) — check the entry."))

    # Simple contradictions
    if vitals.avpu is not None and vitals.mobility is not None:
        if vitals.avpu.value == "unresponsive" and vitals.mobility.value == "walking":
            issues.append(Issue(field="avpu", severity=Severity.WARNING,
                message="Recorded as unresponsive but walking — check AVPU and mobility."))
    if vitals.spo2 is not None and vitals.spo2 >= 97 and vitals.respiratory_rate is not None \
            and vitals.respiratory_rate > 34:
        issues.append(Issue(field="respiratory_rate", severity=Severity.WARNING,
            message="Very fast breathing with normal SpO2 — re-count the breathing rate."))

    ok = not any(i.severity == Severity.ERROR for i in issues)
    return ValidationReport(ok=ok, issues=issues, missing_required=missing)
