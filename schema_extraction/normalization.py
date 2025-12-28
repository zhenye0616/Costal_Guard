from typing import Any, Dict

from .constants import DEFAULT_PROHIBITED_INFERENCES

from .models import StructuredIncident


def normalize_ranges(value: str) -> str:
    return value.replace("\u2013", "-").replace("–", "-")


def apply_deterministic_rules(incident: StructuredIncident) -> StructuredIncident:
    data: Dict[str, Any] = incident.model_dump()

    # Ensure prohibited_inferences is populated (policy guardrail).
    if not data.get("prohibited_inferences"):
        data["prohibited_inferences"] = DEFAULT_PROHIBITED_INFERENCES

    # Coerce "unknown" strings to booleans for required boolean fields while preserving uncertainty.
    obs = data.get("observations", {})
    uncertainties = set(data.get("uncertainties", []))
    if isinstance(obs.get("persons_visible"), str):
        obs["persons_visible"] = False
        uncertainties.add("persons_visible")
    if isinstance(obs.get("debris_observed"), str):
        obs["debris_observed"] = False
        uncertainties.add("debris_observed")
    data["observations"] = obs

    env = data.get("environment", {})
    if "sea_state_ft" in env and isinstance(env["sea_state_ft"], str):
        env["sea_state_ft"] = normalize_ranges(env["sea_state_ft"])
    wind = env.get("wind", {})
    if isinstance(wind, dict) and "speed_kts" in wind and isinstance(wind["speed_kts"], str):
        wind["speed_kts"] = normalize_ranges(wind["speed_kts"])
    data["environment"] = env

    control = data.get("control_state", {})
    if obs.get("debris_observed") and control.get("stability_compromised") is False:
        control["stability_compromised"] = True
    data["control_state"] = control

    uncertainties = set(uncertainties or data.get("uncertainties", []))
    if env.get("water_temp_f") is not None:
        try:
            temp_val = float(env["water_temp_f"])
        except (TypeError, ValueError):
            temp_val = None
        if temp_val is not None and temp_val < 60:
            if "persons_in_water" in uncertainties or data["incident_classification"]["case_type"] == "person_in_water":
                data["human_risk_context"]["hypothermia_risk"] = True

    # Confidence downgrade guard: if high but corroborating signals < 2, downgrade to medium and note uncertainty.
    ic = data.get("incident_classification", {})
    corroborating_signals = int(bool(obs.get("debris_observed"))) + int(bool(obs.get("persons_visible")))
    conf = ic.get("confidence")
    if conf == "high" and corroborating_signals < 2:
        ic["confidence"] = "medium"
        uncertainties.add("confidence_downgraded_due_to_insufficient_corroboration")
        data["incident_classification"] = ic
    if conf not in {"high", "medium", "low"}:
        ic["confidence"] = "low"
        uncertainties.add("confidence_inferred_low")
        data["incident_classification"] = ic

    # Coerce offshore if non-boolean
    loc = data.get("location_context", {})
    if isinstance(loc.get("offshore"), str):
        loc["offshore"] = False
        uncertainties.add("offshore")
        data["location_context"] = loc

    # Coerce hypothermia_risk if non-boolean
    hr = data.get("human_risk_context", {})
    if isinstance(hr.get("hypothermia_risk"), str):
        hr["hypothermia_risk"] = False
        uncertainties.add("hypothermia_risk")
        data["human_risk_context"] = hr

    # Timezone uncertainty if report_time_local missing TZ
    time_ctx = data.get("time_context", {})
    if isinstance(time_ctx.get("report_time_local"), str):
        if " " not in time_ctx["report_time_local"]:
            uncertainties.add("report_time_timezone")

    data["uncertainties"] = sorted(uncertainties)
    data["prohibited_inferences"] = sorted(set(data.get("prohibited_inferences", [])))

    return StructuredIncident.model_validate(data)
