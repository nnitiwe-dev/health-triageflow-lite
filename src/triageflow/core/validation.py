"""Input validation for triage observations.

Separates *data quality* problems (missing, implausible, stale, contradictory, wrong-unit)
from *scoring*. The agent calls this first and must resolve or surface issues before a
priority is proposed — an unsourced or unsafe flag is worse than no flag.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel

from .models import VitalSigns
from .scoring import REQUIRED_TEWS_PARAMS

# Plausible physiological ranges for adults (outside = almost certainly an entry error).
PLAUSIBLE = {
    "respiratory_rate": (4, 60),
    "heart_rate": (20, 250),
    "systolic_bp": (40, 300),
    "temperature_c": (28.0, 43.0),
    "spo2": (40, 100),
    "glucose_mmol": (0.5, 50),
    "diastolic_bp": (20, 200),
    "weight_kg": (1, 400),
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


def validate_vitals(vitals: VitalSigns, now: datetime | None = None) -> ValidationReport:
    now = now or datetime.now(UTC)
    issues: list[Issue] = []
    missing: list[str] = []

    for param in REQUIRED_TEWS_PARAMS:
        if getattr(vitals, param) is None:
            missing.append(param)

    # Plausibility / unit sanity
    for field, (lo, hi) in PLAUSIBLE.items():
        val = getattr(vitals, field)
        if val is None:
            continue
        if field == "temperature_c" and val > 45:
            issues.append(Issue(field=field, severity=Severity.ERROR,
                message=f"Temperature {val} looks like Fahrenheit — enter °C."))
        elif not (lo <= val <= hi):
            issues.append(Issue(field=field, severity=Severity.ERROR,
                message=f"{field}={val} is outside the plausible range {lo}-{hi}; check entry."))

    # Staleness
    if vitals.measured_at is not None:
        age_min = (now - vitals.measured_at).total_seconds() / 60
        if age_min > STALE_MINUTES:
            issues.append(Issue(field="measured_at", severity=Severity.WARNING,
                message=f"Vitals measured {int(age_min)} min ago (> {STALE_MINUTES}); consider re-check."))

    # BP sanity: diastolic should be below systolic
    if vitals.diastolic_bp is not None and vitals.systolic_bp is not None \
            and vitals.diastolic_bp >= vitals.systolic_bp:
        issues.append(Issue(field="diastolic_bp", severity=Severity.ERROR,
            message=f"Diastolic ({vitals.diastolic_bp}) ≥ systolic ({vitals.systolic_bp}); check entry."))

    # Simple contradictions
    if vitals.avpu is not None and vitals.mobility is not None:
        if vitals.avpu.value == "unresponsive" and vitals.mobility.value == "walking":
            issues.append(Issue(field="avpu", severity=Severity.WARNING,
                message="Unresponsive but recorded as walking — verify AVPU/mobility."))
    if vitals.spo2 is not None and vitals.spo2 >= 97 and vitals.respiratory_rate is not None \
            and vitals.respiratory_rate > 34:
        issues.append(Issue(field="respiratory_rate", severity=Severity.WARNING,
            message="Very high RR with normal SpO2 — confirm RR count."))

    ok = not any(i.severity == Severity.ERROR for i in issues)
    return ValidationReport(ok=ok, issues=issues, missing_required=missing)
