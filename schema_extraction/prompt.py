from typing import Any, List

from .constants import (
    CASE_TYPES,
    CONFIDENCE_VALUES,
    NAV_ZONES,
    PROPULSION_VALUES,
    RISK_TYPES,
    TIME_SINCE_VALUES,
    TIMEZONE_ABBREVIATIONS,
    VISIBILITY_VALUES,
    WAVE_PERIOD_VALUES,
)


def normalize_whitespace(text: str) -> str:
    return " ".join(text.replace("\u2013", "-").split()).strip()


def stringify_json(value: Any, prefix: str = "") -> List[str]:
    lines: List[str] = []
    if isinstance(value, dict):
        for k, v in value.items():
            lines.extend(stringify_json(v, f"{prefix}{k}: "))
    elif isinstance(value, list):
        for item in value:
            lines.extend(stringify_json(item, f"{prefix}- "))
    else:
        lines.append(f"{prefix}{value}")
    return lines


def normalize_to_text(raw_input: Any, source_format: str) -> str:
    if source_format != "txt":
        raise ValueError("source_format must be 'txt'")
    return normalize_whitespace(str(raw_input))


def build_prompt(normalized_text: str) -> str:
    system_instruction = (
        "You are a STRICT information extraction system. "
        "Your task is to extract ONLY facts explicitly stated in the incident report. "
        "Do NOT infer, deduce, assume, interpret, classify, or derive any information. "
        "Do NOT use world knowledge. "
        "If a field is not explicitly stated, set it to \"unknown\" or null as specified. "
        "Output ONLY valid JSON that exactly matches the provided schema."
    )

    schema_definition = "\n".join(
        [
            "Schema:",
            "incident_classification: { case_type one of "
            + str(sorted(CASE_TYPES))
            + ", confidence one of "
            + str(sorted(CONFIDENCE_VALUES))
            + " | \"unknown\" }",
            'time_context: { report_time_local string ("HH:MM TZ" where TZ in '
            + str(sorted(TIMEZONE_ABBREVIATIONS))
            + ' OR "HH:MM"), time_since_incident one of '
            + str(sorted(TIME_SINCE_VALUES))
            + " | \"unknown\" }",
            "location_context: { lat number|null, lon number|null, relative_description string|\"unknown\", offshore boolean|\"unknown\", distance_from_shore_nm number|null, "
            + "navigational_zone one of "
            + str(sorted(NAV_ZONES))
            + " | \"unknown\" }",
            "control_state: { maneuverable yes|no|\"unknown\", stability_compromised boolean|\"unknown\", propulsion_status one of "
            + str(sorted(PROPULSION_VALUES))
            + " | \"unknown\" }",
            "environment: { sea_state_ft string|\"unknown\", wave_period one of "
            + str(sorted(WAVE_PERIOD_VALUES))
            + " | \"unknown\", wind { direction string|\"unknown\", speed_kts string|\"unknown\", descriptor string|null }, visibility one of "
            + str(sorted(VISIBILITY_VALUES))
            + " | \"unknown\", water_temp_f number|null }",
            "observations: { debris_observed boolean|\"unknown\", debris_types [string], persons_visible boolean|\"unknown\" }",
            'human_risk_context: { occupant_count integer|\"unknown\", children_present boolean|\"unknown\", hypothermia_risk boolean|\"unknown\" }',
            "uncertainties: list of strings",
            "prohibited_inferences: list of strings",
        ]
    )

    rules = (
        "Rules:\n"
        "- Extract ONLY information explicitly stated in the report.\n"
        "- NEVER infer or derive values from other fields.\n"
        "- NEVER apply conditional logic between fields.\n"
        "- If a value is not explicitly stated, set it to \"unknown\" or null as defined in the schema.\n"
        "- Do NOT normalize, reinterpret, or rephrase facts.\n"
        "- Do NOT assess severity, risk, intent, cause, fault, or safety.\n"
        "- confidence must be copied only if explicitly stated; otherwise set to \"unknown\".\n"
        "- offshore may only be set if explicitly stated.\n"
        "- navigational_zone may only be set if explicitly stated.\n"
        "- persons_visible may only be true or false if explicitly stated; otherwise set to \"unknown\".\n"
        "- debris_observed may only be true or false if explicitly stated; otherwise set to \"unknown\".\n"
        "- hypothermia_risk may only be set if explicitly stated.\n"
        "- uncertainties must list the names of fields that were not explicitly stated.\n"
        "- prohibited_inferences must list prohibited categories not determined, chosen from:\n"
        "  [\"intent\", \"criminality\", \"cause\", \"fault\", \"liability\", \"safety_outcome\", \"risk_assessment\"]\n"
        "- Output JSON ONLY. No prose. No explanations."
    )

    examples = (
        "Examples of valid values:\n"
        '- report_time_local: "16:42 PDT"\n'
        '- report_time_local: "23:36"\n'
        '- visibility: "good" | "fair" | "poor" | "unknown"\n'
        "- persons_visible: true | false | \"unknown\"\n"
        "- debris_observed: true | false | \"unknown\"\n"
        '- wind: { "direction": "unknown", "speed_kts": "unknown", "descriptor": null }\n'
    )

    incident_block = f"Incident Report:\n<<<\n{normalized_text}\n>>>"

    return "\n\n".join(
        [
            "System Instruction:\n" + system_instruction,
            "Schema Definition:\n" + schema_definition,
            "Extraction Rules:\n" + rules,
            "Examples:\n" + examples,
            incident_block,
        ]
    )
