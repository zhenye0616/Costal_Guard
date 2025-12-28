"""
Stage 2 (Temporal Framing) — derive minimal temporal signals from Stage 1 output.

This module:
- Calls Stage 1 extraction (`schema_extraction.pipeline.extract_structured`) when given a raw incident path.
- Accepts already structured JSON as input (no LLM call).
- Emits a small set of temporal signals: active_observation, ongoing_condition, momentary_snapshot.
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
class TemporalSignal:
    signal: str
    confidence: str
    evidence: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _prune_to_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Drop fields not defined in StructuredIncident to tolerate older enriched payloads
    (e.g., primary_risk_vectors) without inferring or using them.
    """
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


def _confidence_from_time_since(time_since: str) -> str:
    if time_since == "immediate":
        return "high"
    if time_since == "minutes":
        return "medium"
    if time_since == "hours":
        return "medium"
    return "low"


def derive_temporal_signals(structured_input: StructuredInput, source_format: str = "txt", model: str = "models/gemini-2.5-flash", mock_response: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Pure function: derive temporal signals from Stage 1 output without inferring causality or duration.

    - active_observation: grounded in immediate/near-immediate reporting or explicit observations.
    - ongoing_condition: grounded in time_since_incident indicating persistence beyond immediate moment.
    - momentary_snapshot: absence of duration cues.
    """
    incident = _load_structured(structured_input, source_format, model, mock_response)
    time_ctx = incident.time_context
    obs = incident.observations

    signals: List[TemporalSignal] = []

    # active_observation
    active_evidence: List[str] = []
    if time_ctx.time_since_incident in {"immediate", "minutes"}:
        active_evidence.append(f"time_since_incident = {time_ctx.time_since_incident}")
    if time_ctx.report_time_local:
        active_evidence.append(f"report_time_local provided ({time_ctx.report_time_local})")
    if obs.debris_observed is True or obs.persons_visible is True:
        active_evidence.append("direct observation recorded in report")
    if active_evidence:
        signals.append(
            TemporalSignal(
                signal="active_observation",
                confidence=_confidence_from_time_since(time_ctx.time_since_incident),
                evidence=active_evidence,
            )
        )

    # ongoing_condition
    ongoing_evidence: List[str] = []
    if time_ctx.time_since_incident in {"minutes", "hours"}:
        ongoing_evidence.append(f"time_since_incident = {time_ctx.time_since_incident}")
    if ongoing_evidence:
        signals.append(
            TemporalSignal(
                signal="ongoing_condition",
                confidence=_confidence_from_time_since(time_ctx.time_since_incident),
                evidence=ongoing_evidence,
            )
        )

    # momentary_snapshot
    momentary_evidence: List[str] = []
    if not ongoing_evidence and time_ctx.time_since_incident == "unknown":
        momentary_evidence.append("time_since_incident = unknown")
    if not ongoing_evidence and not active_evidence:
        momentary_evidence.append("no duration cues present in structured report")
    if momentary_evidence:
        signals.append(
            TemporalSignal(
                signal="momentary_snapshot",
                confidence="low",
                evidence=momentary_evidence,
            )
        )

    return [s.to_dict() for s in signals]


# Alias for simplified pipeline usage
def derive_temporal(structured_input: StructuredInput, source_format: str = "txt", model: str = "models/gemini-2.5-flash", mock_response: Optional[str] = None) -> List[Dict[str, Any]]:
    return derive_temporal_signals(structured_input, source_format=source_format, model=model, mock_response=mock_response)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Derive temporal framing signals from Stage 1 structured output")
    parser.add_argument("source", help="Path to structured JSON (stage 1 output) or raw .txt (stage 1 will run)")
    parser.add_argument("--source-format", default="txt", help="Source format (txt only if running stage 1)")
    parser.add_argument("--model", default="models/gemini-2.5-flash", help="LLM model id (used only if running stage 1)")
    parser.add_argument("--mock-response", help="Path to JSON file to bypass live LLM call for stage 1")
    args = parser.parse_args(argv)

    signals = derive_temporal_signals(
        structured_input=args.source,
        source_format=args.source_format,
        model=args.model,
        mock_response=args.mock_response,
    )
    json.dump({"temporal_signals": signals}, fp=sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
