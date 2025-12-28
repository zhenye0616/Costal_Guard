# ---- Incident Classification ----

CASE_TYPES = {
    "recreational_vessel_distress",
    "disabled_vessel",
    "person_in_water",
    "suspicious_vessel_activity",
    "unknown",
}


# ---- Classification Confidence ----

CONFIDENCE_VALUES = {
    "high",
    "medium",
    "low",
}


# ---- Vessel Control State ----

MANEUVERABLE_VALUES = {
    "yes",
    "no",
    "unknown",
}

PROPULSION_VALUES = {
    "normal",
    "degraded",
    "lost",
    "unknown",
}


# ---- Temporal Context ----

TIME_SINCE_VALUES = {
    "unknown",
    "immediate",   # explicitly documented as "just reported / ongoing"
    "minutes",
    "hours",
}


# ---- Environmental Context ----

WAVE_PERIOD_VALUES = {
    "short",
    "moderate",
    "long",
    "unknown",
}

VISIBILITY_VALUES = {
    "good",
    "fair",
    "poor",
    "unknown",
}


# ---- Navigational Context ----
# IMPORTANT: Must be explicitly stated in report; otherwise use "unknown"

NAV_ZONES = {
    "open_coastal_water",
    "traffic_separation_scheme",
    "harbor",
    "unknown",
}


# ---- Risk Modeling (Non-Intent, Non-Predictive) ----

RISK_SEVERITY = {
    "high",
    "medium",
    "low",
}

RISK_TYPES = {
    "possible_capsize",
    "collision_risk",
    "hypothermia",
    "loss_of_control",
    "suspicious_navigation_pattern",  # replaces "potential_illicit_activity"
    "unknown",
}


# ---- Timezone Handling ----
# Timezone may ONLY be populated if explicitly present in source report

TIMEZONE_ABBREVIATIONS = {
    "UTC",
    "GMT",
    "PST",
    "PDT",
    "MST",
    "MDT",
    "CST",
    "CDT",
    "EST",
    "EDT",
    "AKST",
    "AKDT",
    "HST",
}


# ---- Guardrails: Explicitly Prohibited Inferences ----

DEFAULT_PROHIBITED_INFERENCES = [
    "causes",
    "intent",
    "outcomes",
    "safety",
]
