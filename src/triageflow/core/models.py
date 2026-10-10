"""Core data models for TriageFlow.

Separation of concerns (challenge requirement): raw *observations* (``VitalSigns``,
``Encounter``) are kept distinct from *derived* scores and the *proposed* decision
(``TriageResult``). Nothing here is a diagnosis; the output is a sourced priority a
clinician confirms.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from .sats_rules import AVPU, Mobility, Priority


def _utcnow() -> datetime:
    return datetime.now(UTC)


class VitalSigns(BaseModel):
    """Measured observations at triage. All optional so we can detect what is missing."""

    respiratory_rate: float | None = Field(None, description="breaths/min")
    heart_rate: float | None = Field(None, description="pulse / heart rate, beats/min")
    systolic_bp: float | None = Field(None, description="systolic BP mmHg (used by TEWS)")
    temperature_c: float | None = Field(None, description="degrees Celsius")
    avpu: AVPU | None = Field(None, description="Alert/Voice/Pain/Unresponsive/Confused")
    mobility: Mobility | None = Field(None, description="walking/with_help/immobile")
    trauma: bool | None = Field(None, description="trauma-related presentation")
    # Additional investigations (not part of the 7 TEWS params, but can change priority)
    spo2: float | None = Field(None, description="oxygen saturation %")
    glucose_mmol: float | None = Field(None, description="finger-prick glucose mmol/L")
    # Recorded per real nurse workflow (documentation fidelity); diastolic & weight are NOT
    # scored by adult TEWS — systolic BP is. Pain score maps to the SATS pain discriminator.
    diastolic_bp: float | None = Field(None, description="diastolic BP mmHg (recorded, not scored)")
    weight_kg: float | None = Field(None, description="weight in kg (recorded, not scored in adults)")
    pain_score: int | None = Field(None, ge=0, le=10, description="pain 0-10 → SATS pain discriminator")
    measured_at: datetime = Field(default_factory=_utcnow)


class Encounter(BaseModel):
    """A triage encounter for an adult patient (>12y or >150cm)."""

    encounter_id: str
    age_years: int | None = None
    sex: str | None = None
    presenting_complaint: str = ""
    arrival_time: datetime = Field(default_factory=_utcnow)
    vitals: VitalSigns = Field(default_factory=VitalSigns)
    # Discriminator codes asserted for this patient (from structured input or NL extraction,
    # each confirmed by the nurse before scoring).
    discriminator_codes: list[str] = Field(default_factory=list)
    free_text_notes: str = ""


class Evidence(BaseModel):
    """One sourced reason behind the priority. Every priority must cite entries like these."""

    kind: str                 # "tews" | "discriminator" | "additional_investigation"
    parameter: str            # e.g. "heart_rate" or "chest_pain"
    observed: str             # the value / assertion that triggered it
    points: int | None = None    # TEWS points contributed (None for discriminators)
    band_label: str | None = None
    implies: Priority | None = None  # priority a discriminator implies
    source: str = ""          # citation string


class TriageResult(BaseModel):
    """Derived, proposed priority — NOT confirmed until a clinician approves it."""

    encounter_id: str
    ruleset_version: str
    tews_total: int | None = None
    tews_band: Priority | None = None
    discriminator_level: Priority | None = None
    proposed_priority: Priority | None = None
    target_time: str | None = None
    disposition: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)      # observations needed but absent
    warnings: list[str] = Field(default_factory=list)     # implausible / stale / contradictory
    complete: bool = False     # False => do NOT show a band; needs more data / review
    computed_at: datetime = Field(default_factory=_utcnow)

    def citation_count(self) -> int:
        return len([e for e in self.evidence if e.source])
