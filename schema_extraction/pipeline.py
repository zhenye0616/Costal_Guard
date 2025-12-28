import json
from typing import Any, Dict, Optional

from pydantic import ValidationError

from .constants import DEFAULT_PROHIBITED_INFERENCES
from .llm import call_llm
from .normalization import apply_deterministic_rules
from .prompt import build_prompt, normalize_to_text
from .models import StructuredIncident


def rejected(reason: str) -> Dict[str, str]:
    return {
        "status": "rejected",
        "reason": reason,
        "safe_action": "no_insights_generated",
    }


def extract_structured(
    raw_input: Any,
    source_format: str,
    model: str = "models/gemini-2.5-flash",
    mock_response: Optional[str] = None,
) -> Dict[str, Any]:
    if source_format != "txt":
        return rejected("source_format_must_be_txt")
    normalized_text = normalize_to_text(raw_input, source_format)
    prompt = build_prompt(normalized_text)
    try:
        llm_output = mock_response if mock_response is not None else call_llm(prompt, model)
        parsed = json.loads(llm_output)
        # Phase 1: fill guardrail defaults before strict validation
        if not parsed.get("prohibited_inferences"):
            parsed["prohibited_inferences"] = DEFAULT_PROHIBITED_INFERENCES
        # Pre-downgrade confidence if high without corroborating signals (lenient pre-check)
        try:
            obs_raw = parsed.get("observations", {})
            signals = int(obs_raw.get("debris_observed") is True) + int(obs_raw.get("persons_visible") is True)
            ic_raw = parsed.get("incident_classification", {})
            if ic_raw.get("confidence") == "high" and signals < 2:
                ic_raw["confidence"] = "medium"
                parsed["incident_classification"] = ic_raw
                parsed.setdefault("uncertainties", [])
                if isinstance(parsed["uncertainties"], list):
                    parsed["uncertainties"].append("confidence_downgraded_due_to_insufficient_corroboration")
        except Exception:
            # If pre-check fails, proceed; later validation will catch any structural issues.
            pass
        # Phase 2: initial validation (allows coercions), then deterministic normalization
        partial = StructuredIncident.model_validate(parsed)
        normalized = apply_deterministic_rules(partial)
        # Phase 3: final strict validation (after downgrades/repairs)
        final = StructuredIncident.model_validate(normalized.model_dump())
        return final.model_dump()
    except ValidationError as exc:
        return rejected(f"schema_validation_failed: {exc}")
    except json.JSONDecodeError as exc:
        return rejected(f"invalid_json: {exc}")
    except Exception as exc:  # pylint: disable=broad-except
        return rejected(str(exc))
