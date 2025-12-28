"""
Stage 3 — Scene Simulation (non-inferential, human-facing only).

This module renders the canonical prompt template for hypothetical visualizations.
It accepts only Stage 2 outputs (selected scenario, situational factors, operational
constraints, temporal framing) and emits a text prompt plus the required overlay label.

Rules enforced here:
- Hypothetical only; never evidence or ground truth.
- No inference of intent, risk, or outcomes.
- One scenario at a time; no blending or ranking.
- Constraints are used to preserve ambiguity (silhouettes, partial occlusion, neutral cues).
"""


from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence
from typing import Any, Mapping

OVERLAY_LABEL = "Plausible Scenario Visualization — Hypothetical · Not Evidence · Not Ground Truth"


def _sanitize_list(values: Sequence[str]) -> List[str]:
    """
    Drop empty/None entries and trim whitespace to keep prompts clean.
    """
    return [v.strip() for v in values if v and v.strip()]


@dataclass
class Prompt:
    """
    Contract for prompt construction.
    """

    scenario: str
    situational_factors: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    temporal_context: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.scenario or not str(self.scenario).strip():
            raise ValueError("scenario is required for prompt construction")
        self.scenario = str(self.scenario).strip()
        self.situational_factors = _sanitize_list(self.situational_factors)
        self.constraints = _sanitize_list(self.constraints)
        self.temporal_context = _sanitize_list(self.temporal_context)


def _section_lines(title: str, items: List[str], fallback: str) -> List[str]:
    lines = [f"{title}:"]
    if items:
        lines.extend(f"- {item}" for item in items)
    else:
        lines.append(f"- {fallback}")
    return lines


def build_scene_prompt(inputs: Prompt) -> str:
    """
    Render the canonical Stage 3 prompt from Stage 2 outputs.
    """
    lines: List[str] = [
        "You are generating a hypothetical visualization for cognitive illustration only.",
        "",
        "This visualization corresponds to a plausible scenario, not a confirmed event.",
        "It is not evidence and not ground truth.",
        "Do not infer intent, risk, or outcomes.",
        "Avoid specific identifying details.",
        "",
    ]

    lines.extend(_section_lines("Scenario", [inputs.scenario], "no scenario selected"))
    lines.append("")
    lines.extend(_section_lines("Situational context", inputs.situational_factors, "none provided"))
    lines.append("")
    lines.extend(_section_lines("Operational constraints", inputs.constraints, "none provided"))
    lines.append("")
    lines.extend(_section_lines("Temporal framing", inputs.temporal_context, "none provided"))
    lines.append("")

    lines.extend(
        [
            "Visual guidance:",
            "- generic elements only",
            "- ambiguous details where constraints apply",
            "- neutral posture",
            "- no distress or threat indicators",
            "- apply constraints as ambiguity drivers (silhouettes, partial occlusion, neutral cues)",
            "",
            "Style:",
            "- subdued",
            "- observational",
            "- non-dramatic",
        ]
    )

    return "\n".join(lines)


def build_visualization_request(inputs: Prompt) -> dict:
    """
    Convenience wrapper that packages both the prompt and the required overlay label.
    """
    return {
        "prompt": build_scene_prompt(inputs),
        "overlay_label": OVERLAY_LABEL,
    }


def build_visualization_request_from_analyze(analyze_payload: Mapping[str, Any], scenario: str) -> dict:
    """
    Build a Stage 3 visualization request directly from a saved /analyze JSON payload.
    """
    scenario_names = [s.get("scenario") for s in analyze_payload.get("scenarios", []) if isinstance(s, Mapping)]
    if scenario not in scenario_names:
        raise ValueError("selected scenario must be present in analyze_payload['scenarios']")

    inputs = Prompt(
        scenario=scenario,
        situational_factors=[
            f.get("factor", "") for f in analyze_payload.get("situational_factors", []) if isinstance(f, Mapping)
        ],
        constraints=[
            c.get("constraint", "") for c in analyze_payload.get("operational_constraints", []) if isinstance(c, Mapping)
        ],
        temporal_context=[t.get("signal", "") for t in analyze_payload.get("temporal", []) if isinstance(t, Mapping)],
    )
    return build_visualization_request(inputs)


__all__ = [
    "Prompt",
    "OVERLAY_LABEL",
    "build_scene_prompt",
    "build_visualization_request",
    "build_visualization_request_from_analyze",
]
