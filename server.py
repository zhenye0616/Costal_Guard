import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from schema_extraction.pipeline import extract_structured
from incident_analysis.temporal import derive_temporal
from incident_analysis.situational import derive_factors
from incident_analysis.operational_constraints import derive_constraints
from incident_analysis.scenarios import derive_scenarios
from nanobanana.prompt import build_visualization_request_from_analyze
from nanobanana.visual import build_spatial_context_from_payload
from nanobanana.models import normalize_image_model
from utils import (
    call_image_generation,
    extract_lat_lon,
    load_cached_structured,
    load_incident_json,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(title="USCG Incident Extraction API", version="1.0.0")

logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Request Models
# -----------------------------------------------------------------------------

class ExtractRequest(BaseModel):
    source_path: str


class RawRequest(BaseModel):
    source_path: str


class NanoBananaRequest(BaseModel):
    source_path: str
    scenario: str
    model: str | None = None  # ← allow legacy / explicit model requests


# Routes
# -----------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/raw")
def raw_text(source_path: str) -> Any:
    text, _ = load_incident_json(source_path)
    return {"source_path": source_path, "text": text}


@app.post("/extract")
def extract(req: ExtractRequest) -> Any:
    payload, fmt = load_incident_json(req.source_path)
    result = extract_structured(payload, source_format=fmt)
    return {"stage": "stage_1_extraction", "source_path": req.source_path, "result": result}


@app.post("/analyze")
def analyze(req: ExtractRequest) -> Any:
    payload, fmt = load_incident_json(req.source_path)
    structured = extract_structured(payload, source_format=fmt)

    if isinstance(structured, dict) and structured.get("status") == "rejected":
        cached = load_cached_structured(req.source_path)
        if cached is None:
            return {"stage": "stage_1_extraction", "structured": structured}
        structured = cached

    temporal = derive_temporal(structured)
    factors = derive_factors(structured)
    constraints = derive_constraints(structured)
    scenarios = derive_scenarios(
        structured_input=structured,
        temporal_signals=temporal,
        situational_factors=factors,
        constraints=constraints,
    )

    return {
        "stage": "stage_2_analysis",
        "structured": structured,
        "temporal": temporal,
        "situational_factors": factors,
        "operational_constraints": constraints,
        "scenarios": scenarios,
    }


@app.post("/nanobanana/generate")
def nanobanana_generate(req: NanoBananaRequest) -> Any:
    payload, fmt = load_incident_json(req.source_path)
    structured = extract_structured(payload, source_format=fmt)

    if isinstance(structured, dict) and structured.get("status") == "rejected":
        cached = load_cached_structured(req.source_path)
        if cached is None:
            raise HTTPException(status_code=422, detail="stage_1_extraction_rejected")
        structured = cached

    temporal = derive_temporal(structured)
    factors = derive_factors(structured)
    constraints = derive_constraints(structured)
    scenarios = derive_scenarios(
        structured_input=structured,
        temporal_signals=temporal,
        situational_factors=factors,
        constraints=constraints,
    )

    if req.scenario not in [s.get("scenario") for s in scenarios if isinstance(s, dict)]:
        raise HTTPException(status_code=400, detail="Invalid scenario selection")

    prompt_bundle = build_visualization_request_from_analyze(
        {
            "situational_factors": factors,
            "operational_constraints": constraints,
            "temporal": temporal,
            "scenarios": scenarios,
        },
        scenario=req.scenario,
    )
    logger.info("Prompt bundle generated", extra={"prompt_bundle": prompt_bundle})

    lat, lon = extract_lat_lon(structured)
    spatial = build_spatial_context_from_payload(
        {
            "lat": lat,
            "lon": lon,
            "situational_factors": [f.get("factor", "") for f in factors if isinstance(f, dict)],
            "temporal_context": [t.get("signal", "") for t in temporal if isinstance(t, dict)],
        }
    )
    logger.info("Spatial context generated", extra={"spatial": spatial})

    image_urls = [
        img.get("url", "")
        for img in spatial.get("images", [])
        if isinstance(img, dict) and img.get("url")
    ]

    requested_model = req.model or "gemini-3-pro-image-preview"
    model, family = normalize_image_model(requested_model)

    gemini_images = call_image_generation(
        model=model,
        family=family,
        prompt=prompt_bundle["prompt"],
        context_note=spatial.get("context_note", ""),
        image_urls=image_urls,
    )

    return {
        "stage": "stage_3_scene_simulation",
        "selected_scenario": req.scenario,
        "prompt": prompt_bundle["prompt"],
        "overlay_label": prompt_bundle["overlay_label"],
        "spatial_context": spatial,
        "image_model_requested": requested_model,
        "image_model_used": model,
        "image_model_family": family,
        "gemini_images": gemini_images,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
