from typing import List, Union, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .constants import (
    CASE_TYPES,
    CONFIDENCE_VALUES,
    MANEUVERABLE_VALUES,
    NAV_ZONES,
    PROPULSION_VALUES,
    TIME_SINCE_VALUES,
    TIMEZONE_ABBREVIATIONS,
    VISIBILITY_VALUES,
    WAVE_PERIOD_VALUES,
)

# -------------------------
# Incident Classification
# -------------------------

class IncidentClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_type: str
    confidence: str  # low | medium | high | unknown

    @field_validator("case_type")
    @classmethod
    def validate_case_type(cls, v: str) -> str:
        if v not in CASE_TYPES:
            raise ValueError("invalid case_type")
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: str) -> str:
        if v not in CONFIDENCE_VALUES and v != "unknown":
            raise ValueError("invalid confidence")
        return v


# -------------------------
# Time Context
# -------------------------

class TimeContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_time_local: str
    time_since_incident: str  # may be "unknown"

    @field_validator("time_since_incident")
    @classmethod
    def validate_time_since(cls, v: str) -> str:
        if v not in TIME_SINCE_VALUES:
            raise ValueError("invalid time_since_incident")
        return v

    @field_validator("report_time_local")
    @classmethod
    def validate_report_time_local(cls, v: str) -> str:
        parts = v.split()
        if len(parts) == 2:
            time_part, tz = parts
            if tz not in TIMEZONE_ABBREVIATIONS:
                raise ValueError("invalid timezone abbreviation")
        elif len(parts) == 1:
            time_part = parts[0]
        else:
            raise ValueError("report_time_local must be 'HH:MM TZ' or 'HH:MM'")

        if ":" in time_part:
            h, m = time_part.split(":")
        elif len(time_part) == 4:
            h, m = time_part[:2], time_part[2:]
        else:
            raise ValueError("invalid time format")

        if not (0 <= int(h) <= 23 and 0 <= int(m) <= 59):
            raise ValueError("time out of range")

        return v


# -------------------------
# Environment
# -------------------------

class Wind(BaseModel):
    model_config = ConfigDict(extra="ignore")

    direction: str = "unknown"
    speed_kts: str = "unknown"


class Environment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sea_state_ft: str
    wave_period: str
    wind: Wind = Field(default_factory=Wind)
    visibility: str
    water_temp_f: Union[int, float, None] = None

    @field_validator("wave_period")
    @classmethod
    def validate_wave_period(cls, v: str) -> str:
        if v not in WAVE_PERIOD_VALUES:
            raise ValueError("invalid wave_period")
        return v

    @field_validator("visibility")
    @classmethod
    def validate_visibility(cls, v: str) -> str:
        if v not in VISIBILITY_VALUES:
            raise ValueError("invalid visibility")
        return v


# -------------------------
# Location Context
# -------------------------

class LocationContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lat: float
    lon: float
    relative_description: str
    offshore: Union[bool, Literal["unknown"]]
    distance_from_shore_nm: float
    navigational_zone: Union[str, Literal["unknown"]]

    @field_validator("navigational_zone")
    @classmethod
    def validate_zone(cls, v: str) -> str:
        if v != "unknown" and v not in NAV_ZONES:
            raise ValueError("invalid navigational_zone")
        return v

    @model_validator(mode="after")
    def check_distance_vs_offshore(self):
        if self.offshore is True and self.distance_from_shore_nm < 1:
            raise ValueError(
                "offshore=true inconsistent with distance_from_shore_nm < 1"
            )
        return self


# -------------------------
# Control State
# -------------------------

class ControlState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    maneuverable: Union[str, Literal["unknown"]]
    stability_compromised: Union[bool, Literal["unknown"]]
    propulsion_status: Union[str, Literal["unknown"]]

    @field_validator("maneuverable")
    @classmethod
    def validate_maneuverable(cls, v: str) -> str:
        if v != "unknown" and v not in MANEUVERABLE_VALUES:
            raise ValueError("invalid maneuverable")
        return v

    @field_validator("propulsion_status")
    @classmethod
    def validate_propulsion(cls, v: str) -> str:
        if v != "unknown" and v not in PROPULSION_VALUES:
            raise ValueError("invalid propulsion_status")
        return v

    @model_validator(mode="after")
    def check_propulsion_vs_maneuverable(self) -> "ControlState":
        if (
            self.propulsion_status == "lost"
            and self.maneuverable == "yes"
        ):
            raise ValueError("lost propulsion incompatible with maneuverable=yes")
        return self


# -------------------------
# Observations
# -------------------------

class Observations(BaseModel):
    model_config = ConfigDict(extra="forbid")

    debris_observed: Union[bool, Literal["unknown"]]
    debris_types: List[str]
    persons_visible: Union[bool, Literal["unknown"]]

    @model_validator(mode="after")
    def check_debris_consistency(self) -> "Observations":
        if self.debris_observed is True and not self.debris_types:
            raise ValueError("debris_types must be non-empty if debris_observed=true")
        return self


# -------------------------
# Human Risk Context (Observed Only)
# -------------------------

class HumanRiskContext(BaseModel):
    """
    hypothermia_risk may ONLY be true if explicitly stated
    or undeniable (e.g., confirmed persons in water + cold temp).
    """
    model_config = ConfigDict(extra="forbid")

    occupant_count: Union[int, Literal["unknown"]]
    children_present: Union[bool, Literal["unknown"]]
    hypothermia_risk: Union[bool, Literal["unknown"]]

    @field_validator("occupant_count")
    @classmethod
    def validate_occupant_count(cls, v):
        if isinstance(v, str) and v != "unknown":
            raise ValueError("occupant_count must be int or 'unknown'")
        return v

    @field_validator("children_present")
    @classmethod
    def validate_children_present(cls, v):
        if isinstance(v, str) and v != "unknown":
            raise ValueError("children_present must be boolean or 'unknown'")
        return v

    @field_validator("hypothermia_risk")
    @classmethod
    def validate_hypothermia_risk(cls, v):
        if isinstance(v, str) and v != "unknown":
            raise ValueError("hypothermia_risk must be boolean or 'unknown'")
        return v


# -------------------------
# Structured Incident (Observed)
# -------------------------

class StructuredIncident(BaseModel):
    """
    OBSERVED FACTS ONLY.
    Allows explicit 'unknown' where reality is uncertain.
    NO derived interpretation.
    """
    model_config = ConfigDict(extra="forbid")

    incident_classification: IncidentClassification
    time_context: TimeContext
    location_context: LocationContext
    control_state: ControlState
    environment: Environment
    observations: Observations
    human_risk_context: HumanRiskContext
    uncertainties: List[str]
    prohibited_inferences: List[str]

    @model_validator(mode="after")
    def ensure_epistemic_constraints(self) -> "StructuredIncident":
        if not self.uncertainties:
            raise ValueError("uncertainties must be non-empty")

        # Confidence sanity check (only when confidence is asserted)
        if self.incident_classification.confidence != "unknown":
            obs = self.observations
            corroboration = int(obs.debris_observed is True) + int(obs.persons_visible is True)
            if self.incident_classification.confidence == "high" and corroboration < 2:
                raise ValueError(
                    "high confidence requires multiple corroborating indicators"
                )

        return self
