"""
Stage 2.2 — Situational Factors

Derive a small, controlled set of situational factors from Stage 1 structured output
and optional temporal signals. No raw text, no intent, no scenarios.
"""

from __future__ import annotations

import argparse
import json
import re
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
class SituationalFactor:
    factor: str
    severity: str  # low | medium | high (impact if relevant, not likelihood)
    confidence: str  # low | medium | high
    evidence: List[str]

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


def _parse_hour(report_time_local: str) -> Optional[int]:
    """
    Extract hour (0-23) from 'HH:MM' or 'HHMM' with optional TZ suffix.
    """
    if not report_time_local:
        return None
    parts = report_time_local.split()
    time_part = parts[0]
    match = re.match(r"^(\d{1,2}):?(\d{2})$", time_part)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    if 0 <= hour <= 23 and 0 <= minute <= 59:
        return hour
    return None


def _is_night(hour: Optional[int]) -> bool:
    return hour is not None and (hour < 6 or hour >= 18)


def _visibility_good(visibility: str) -> Optional[bool]:
    if visibility is None:
        return None
    vis = str(visibility).lower()
    if vis == "unknown":
        return None
    if vis == "good":
        return True
    if vis in {"fair", "poor"}:
        return False
    return None


def derive_situational_factors(structured_input: StructuredInput, source_format: str = "txt", model: str = "models/gemini-2.5-flash", mock_response: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Derive a controlled set of situational factors without inference beyond grounded fields.
    Emits up to: nighttime_operations, harbor_approach, offshore_open_water, nearshore_open_water, good_visibility, sensor_limited_assessment.
    """
    incident = _load_structured(structured_input, source_format, model, mock_response)

    loc = incident.location_context
    env = incident.environment
    time_ctx = incident.time_context

    factors: List[SituationalFactor] = []

    # nighttime_operations
    hour = _parse_hour(time_ctx.report_time_local)
    if _is_night(hour):
        evidence = [f"report_time_local = {time_ctx.report_time_local}"]
        factors.append(
            SituationalFactor(
                factor="nighttime_operations",
                severity="medium",
                confidence="high" if hour is not None else "low",
                evidence=evidence,
            )
        )

    # harbor_approach
    harbor_evidence: List[str] = []
    if loc.navigational_zone == "harbor":
        harbor_evidence.append("navigational_zone = harbor")
    if loc.distance_from_shore_nm is not None and loc.distance_from_shore_nm < 1:
        harbor_evidence.append(f"distance_from_shore_nm = {loc.distance_from_shore_nm}")
    if harbor_evidence:
        factors.append(
            SituationalFactor(
                factor="harbor_approach",
                severity="medium",
                confidence="high" if "navigational_zone = harbor" in harbor_evidence else "medium",
                evidence=harbor_evidence,
            )
        )

    # good_visibility
    vis_good = _visibility_good(env.visibility)
    if vis_good is True:
        factors.append(
            SituationalFactor(
                factor="good_visibility",
                severity="low",
                confidence="high",
                evidence=[f"visibility = {env.visibility}"],
            )
        )

    # spatial context: offshore vs nearshore (mutually exclusive)
    offshore_evidence: List[str] = []
    nearshore_evidence: List[str] = []
    if loc.offshore is True:
        offshore_evidence.append("offshore = true")
    elif loc.offshore is False:
        nearshore_evidence.append("offshore = false")

    if loc.distance_from_shore_nm is not None:
        if loc.distance_from_shore_nm > 1:
            offshore_evidence.append(f"distance_from_shore_nm = {loc.distance_from_shore_nm}")
        else:
            nearshore_evidence.append(f"distance_from_shore_nm = {loc.distance_from_shore_nm}")

    if loc.navigational_zone and loc.navigational_zone != "unknown":
        # Use navigational zone as contextual evidence without overriding offshore flag
        if loc.navigational_zone == "harbor":
            nearshore_evidence.append("navigational_zone = harbor")
        else:
            offshore_evidence.append(f"navigational_zone = {loc.navigational_zone}")

    if offshore_evidence and not nearshore_evidence:
        factors.append(
            SituationalFactor(
                factor="offshore_open_water",
                severity="medium",
                confidence="high" if "offshore = true" in offshore_evidence and any("distance_from_shore_nm" in ev for ev in offshore_evidence) else "medium",
                evidence=offshore_evidence,
            )
        )
    elif nearshore_evidence:
        factors.append(
            SituationalFactor(
                factor="nearshore_open_water",
                severity="medium",
                confidence="high" if "offshore = false" in nearshore_evidence or any("distance_from_shore_nm" in ev for ev in nearshore_evidence) else "medium",
                evidence=nearshore_evidence,
            )
        )

    # sensor_limited_assessment
    sensor_evidence: List[str] = []
    if env.visibility in {"fair", "poor"}:
        sensor_evidence.append(f"visibility = {env.visibility}")
    if sensor_evidence:
        factors.append(
            SituationalFactor(
                factor="sensor_limited_assessment",
                severity="medium",
                confidence="high" if env.visibility in {"fair", "poor"} else "low",
                evidence=sensor_evidence,
            )
        )

    return [f.to_dict() for f in factors]


# Alias for simplified pipeline usage
def derive_factors(structured_input: StructuredInput, source_format: str = "txt", model: str = "models/gemini-2.5-flash", mock_response: Optional[str] = None) -> List[Dict[str, Any]]:
    return derive_situational_factors(structured_input, source_format=source_format, model=model, mock_response=mock_response)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Derive situational factors from Stage 1 structured output")
    parser.add_argument("source", help="Path to structured JSON (stage 1 output) or raw .txt (stage 1 will run)")
    parser.add_argument("--source-format", default="txt", help="Source format (txt only if running stage 1)")
    parser.add_argument("--model", default="models/gemini-2.5-flash", help="LLM model id (used only if running stage 1)")
    parser.add_argument("--mock-response", help="Path to JSON file to bypass live LLM call for stage 1")
    args = parser.parse_args(argv)

    factors = derive_situational_factors(
        structured_input=args.source,
        source_format=args.source_format,
        model=args.model,
        mock_response=args.mock_response,
    )
    json.dump({"situational_factors": factors}, fp=sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
