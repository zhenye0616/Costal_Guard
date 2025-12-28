"""
Stage 2.3 — Operational Constraints

Surface explicit limits on what can be reliably known from current observations.
Non-decisional. No recommendations. Purely epistemic bounds.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

# Allow running as a script without installing as a package
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from schema_extraction.io_utils import load_mock, load_source  # type: ignore  # noqa: E402
from schema_extraction.models import StructuredIncident  # type: ignore  # noqa: E402
from schema_extraction.pipeline import extract_structured  # type: ignore  # noqa: E402

StructuredInput = Union[Dict[str, Any], StructuredIncident, str, Path]


@dataclass
class OperationalConstraint:
    constraint: str
    confidence: str  # low | medium | high (certainty the limitation exists)
    evidence: List[str]
    impact_scope: str  # e.g., human_safety, response_scale, vessel_capability, situation_evolution, attribution_and_intent

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _prune_to_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    allowed = set(StructuredIncident.model_fields.keys())
    return {k: v for k, v in data.items() if k in allowed}


def _load_structured(payload: StructuredInput, source_format: str, model: str, mock_response: Optional[str]) -> StructuredIncident:
    """
    Accept a dict/StructuredIncident, structured JSON path, or raw txt path.
    If a .txt path is provided, Stage 1 extraction is invoked.
    """
    if isinstance(payload, StructuredIncident):
        return payload
    if isinstance(payload, dict):
        return StructuredIncident.model_validate(_prune_to_schema(payload))

    path = Path(payload).expanduser().resolve()
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return StructuredIncident.model_validate(_prune_to_schema(data))

    raw, fmt = load_source(str(path), explicit_format=source_format)
    mock_payload = load_mock(mock_response) if mock_response else None
    result = extract_structured(raw, source_format=fmt, model=model, mock_response=mock_payload)
    if isinstance(result, dict) and result.get("status") == "rejected":
        raise ValueError(f"stage_1_extraction_rejected: {result.get('reason')}")
    return StructuredIncident.model_validate(_prune_to_schema(result))


def derive_operational_constraints(structured_input: StructuredInput, source_format: str = "txt", model: str = "models/gemini-2.5-flash", mock_response: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Derive a minimal set of operational constraints from Stage 1 output.
    Emits up to: unknown_occupant_count, unknown_vessel_capability, unknown_incident_duration, sensor_limited_assessment.
    """
    incident = _load_structured(structured_input, source_format, model, mock_response)

    constraints: List[OperationalConstraint] = []

    # unknown_occupant_count
    occ_evidence: List[str] = []
    if incident.human_risk_context.occupant_count == "unknown":
        occ_evidence.append("occupant_count = unknown")
    if incident.observations.persons_visible in ("unknown", False):
        occ_evidence.append(f"persons_visible = {incident.observations.persons_visible}")
    if occ_evidence:
        constraints.append(
            OperationalConstraint(
                constraint="unknown_occupant_count",
                confidence="high",
                evidence=occ_evidence,
                impact_scope="human_safety",
            )
        )

    # unknown_vessel_capability
    vessel_evidence: List[str] = []
    cs = incident.control_state
    if cs.maneuverable == "unknown":
        vessel_evidence.append("maneuverable = unknown")
    if cs.propulsion_status == "unknown":
        vessel_evidence.append("propulsion_status = unknown")
    if cs.stability_compromised == "unknown":
        vessel_evidence.append("stability_compromised = unknown")
    if vessel_evidence:
        constraints.append(
            OperationalConstraint(
                constraint="unknown_vessel_capability",
                confidence="high",
                evidence=vessel_evidence,
                impact_scope="vessel_capability",
            )
        )

    # unknown_incident_duration
    duration_evidence: List[str] = []
    if incident.time_context.time_since_incident == "unknown":
        duration_evidence.append("time_since_incident = unknown")
    if duration_evidence:
        constraints.append(
            OperationalConstraint(
                constraint="unknown_incident_duration",
                confidence="high",
                evidence=duration_evidence,
                impact_scope="situation_evolution",
            )
        )

    # sensor_limited_assessment (environment-driven)
    sensor_evidence: List[str] = []
    vis = incident.environment.visibility
    if vis in {"fair", "poor", "unknown"}:
        sensor_evidence.append(f"visibility = {vis}")
    if incident.location_context.distance_from_shore_nm is not None and incident.location_context.distance_from_shore_nm > 5:
        sensor_evidence.append(f"distance_from_shore_nm = {incident.location_context.distance_from_shore_nm}")
    if sensor_evidence:
        constraints.append(
            OperationalConstraint(
                constraint="sensor_limited_assessment",
                confidence="high" if vis in {"fair", "poor", "unknown"} else "medium",
                evidence=sensor_evidence,
                impact_scope="situation_evolution",
            )
        )

    return [c.to_dict() for c in constraints]


# Alias for simplified pipeline usage
def derive_constraints(structured_input: StructuredInput, source_format: str = "txt", model: str = "models/gemini-2.5-flash", mock_response: Optional[str] = None) -> List[Dict[str, Any]]:
    return derive_operational_constraints(structured_input, source_format=source_format, model=model, mock_response=mock_response)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Derive operational constraints from Stage 1 structured output")
    parser.add_argument("source", help="Path to structured JSON (stage 1 output) or raw .txt (stage 1 will run)")
    parser.add_argument("--source-format", default="txt", help="Source format (txt only if running stage 1)")
    parser.add_argument("--model", default="models/gemini-2.5-flash", help="LLM model id (used only if running stage 1)")
    parser.add_argument("--mock-response", help="Path to JSON file to bypass live LLM call for stage 1")
    args = parser.parse_args(argv)

    constraints = derive_operational_constraints(
        structured_input=args.source,
        source_format=args.source_format,
        model=args.model,
        mock_response=args.mock_response,
    )
    json.dump({"operational_constraints": constraints}, fp=sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
