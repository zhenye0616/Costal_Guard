import argparse
import json
import sys

from pydantic import ValidationError
from .io_utils import load_mock, load_source
from .models import StructuredIncident
from .normalization import apply_deterministic_rules
from .pipeline import extract_structured, rejected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Incident Report → Structured Operational Schema (v1)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="Run normalization + LLM extraction + validation")
    extract.add_argument("source", help="Path to .txt input")
    extract.add_argument("--source-format", choices=["txt"], default="txt", help="Source format (txt only)")
    extract.add_argument("--model", default="models/gemini-2.5-flash", help="LLM model id (default: gemini-2.5-flash)")
    extract.add_argument("--mock-response", help="Path to JSON file to bypass live LLM call")
    validate = sub.add_parser("validate", help="Validate a structured JSON file")
    validate.add_argument("structured_path", help="Path to structured JSON payload")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "extract":
        raw, fmt = load_source(args.source, args.source_format)
        mock_payload = load_mock(args.mock_response) if args.mock_response else None
        result = extract_structured(raw, source_format=fmt, model=args.model, mock_response=mock_payload)
        json.dump(result, fp=sys.stdout, indent=2)
        print()
        if isinstance(result, dict) and result.get("status") == "rejected":
            sys.exit(1)
    elif args.command == "validate":
        with open(args.structured_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        try:
            structured = StructuredIncident.model_validate(payload)
            normalized = apply_deterministic_rules(structured)
            json.dump(normalized.model_dump(), fp=sys.stdout, indent=2)
            print()
        except ValidationError as exc:
            json.dump(rejected(f"schema_validation_failed: {exc}"), fp=sys.stdout, indent=2)
            print()
            sys.exit(1)


if __name__ == "__main__":
    main()
