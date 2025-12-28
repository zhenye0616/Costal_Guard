"""
Stage 2.4 — Plausible Scenario Enumeration

Enumerate multiple mutually-plausible, non-ranked scenarios to prevent narrative collapse.
Inputs: Stage 2 temporal signals, situational factors, operational constraints (plus Stage 1 facts via those).
Outputs are non-decisional, uncertainty-preserving interpretations.
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
from incident_analysis.temporal import derive_temporal_signals  # type: ignore  # noqa: E402
from incident_analysis.situational import derive_situational_factors  # type: ignore  # noqa: E402
from incident_analysis.operational_constraints import derive_operational_constraints  # type: ignore  # noqa: E402

StructuredInput = Union[Dict[str, Any], StructuredIncident, str, Path]


@dataclass
class Scenario:
    scenario: str
    confidence: str  # low | medium (plausibility, not likelihood)
    supporting_evidence: List[str]
    bounded_by_constraints: List[str]

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


def derive_scenarios(
    structured_input: StructuredInput,
    source_format: str = "txt",
    model: str = "models/gemini-2.5-flash",
    mock_response: Optional[str] = None,
    temporal_signals: Optional[List[Dict[str, Any]]] = None,
    situational_factors: Optional[List[Dict[str, Any]]] = None,
    constraints: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Derive a minimal set of plausible scenarios (non-ranked, uncertainty-preserving).
    Emits exactly three archetypes:
    - benign_or_routine_navigation
    - non_distress_anomalous_navigation
    - other_explanations_possible
    """
    incident = _load_structured(structured_input, source_format, model, mock_response)
    temporal = temporal_signals if temporal_signals is not None else derive_temporal_signals(structured_input, source_format, model, mock_response)
    situational = situational_factors if situational_factors is not None else derive_situational_factors(structured_input, source_format, model, mock_response)
    constraints = constraints if constraints is not None else derive_operational_constraints(structured_input, source_format, model, mock_response)

    constraint_names = [c.get("constraint", "") for c in constraints if isinstance(c, dict)]
    situational_factors = [f.get("factor", "") for f in situational if isinstance(f, dict)]

    scenarios: List[Scenario] = []

    # benign_or_routine_navigation
    benign_evidence = []
    benign_evidence.extend([f for f in situational_factors if f in {"offshore_open_water", "good_visibility"}])
    scenarios.append(
        Scenario(
            scenario="benign_or_routine_navigation",
            confidence="low",
            supporting_evidence=benign_evidence or ["no explicit distress indicators in structured facts"],
            bounded_by_constraints=constraint_names,
        )
    )

    # non_distress_anomalous_navigation
    anomalous_evidence = ["incident classified as recreational_vessel_distress"]
    if incident.control_state.stability_compromised is False:
        anomalous_evidence.append("no confirmed loss of control")
    scenarios.append(
        Scenario(
            scenario="non_distress_anomalous_navigation",
            confidence="low",
            supporting_evidence=anomalous_evidence or ["structured facts allow non-distress anomalous interpretation"],
            bounded_by_constraints=constraint_names,
        )
    )

    # other_explanations_possible (explicit catch-all to preserve plurality)
    scenarios.append(
        Scenario(
            scenario="other_explanations_possible",
            confidence="low",
            supporting_evidence=["multiple high-confidence operational constraints preserve alternate interpretations"],
            bounded_by_constraints=constraint_names,
        )
    )

    return [s.to_dict() for s in scenarios]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Enumerate plausible scenarios (non-ranked, uncertainty-preserving)")
    parser.add_argument("source", help="Path to structured JSON (stage 1 output) or raw .txt (stage 1 will run)")
    parser.add_argument("--source-format", default="txt", help="Source format (txt only if running stage 1)")
    parser.add_argument("--model", default="models/gemini-2.5-flash", help="LLM model id (used only if running stage 1)")
    parser.add_argument("--mock-response", help="Path to JSON file to bypass live LLM call for stage 1")
    args = parser.parse_args(argv)

    scenarios = derive_scenarios(
        structured_input=args.source,
        source_format=args.source_format,
        model=args.model,
        mock_response=args.mock_response,
    )
    json.dump({"scenarios": scenarios}, fp=sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
