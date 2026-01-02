import json
import logging
import os
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

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
    geocode_location,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(title="USCG Incident Extraction API", version="1.0.0")
BASE_DIR = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/incident_context", StaticFiles(directory=BASE_DIR / "incident_context"), name="incident_context")
app.mount("/structured_outputs", StaticFiles(directory=BASE_DIR / "structured_outputs"), name="structured_outputs")


MAX_TEXT_FILE_SIZE = 1 * 1024 * 1024  # 1 MB
MAX_IMAGE_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}


async def validate_and_read_text_file(file: UploadFile) -> str:
    if not file.filename or not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="incident_text must be a .txt file")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="incident_text file is empty")
    if len(raw) > MAX_TEXT_FILE_SIZE:
        raise HTTPException(status_code=413, detail="incident_text file too large")

    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="incident_text must be UTF-8 encoded")


async def validate_and_convert_image(file: UploadFile) -> Image.Image:
    if not file.content_type or file.content_type.lower() not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="incident_image must be a valid image type")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="incident_image file is empty")
    if len(raw) > MAX_IMAGE_FILE_SIZE:
        raise HTTPException(status_code=413, detail="incident_image file too large")

    try:
        img = Image.open(BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=400, detail="incident_image could not be decoded")

    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def extract_location_query(text: str) -> str | None:
    for line in text.splitlines():
        if ":" in line and line.strip().lower().startswith("location"):
            _, value = line.split(":", 1)
            candidate = value.strip()
            if candidate:
                return candidate
    return None


def append_coordinates(text: str, lat: float, lon: float) -> str:
    return f"{text}\n\nCoordinates: {lat:.6f}, {lon:.6f}\n"


# Routes
# -----------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/process-incident")
async def process_incident(
    incident_text: UploadFile = File(...),
    incident_image: UploadFile | None = File(None),
) -> dict:
    text = await validate_and_read_text_file(incident_text)
    has_incident_image = False
    if incident_image is not None:
        img = await validate_and_convert_image(incident_image)
        has_incident_image = True
        img.close()

    location_query = extract_location_query(text)
    if location_query:
        coords = geocode_location(location_query)
        if coords is not None:
            text = append_coordinates(text, coords[0], coords[1])

    structured = extract_structured(text, source_format="txt")
    if isinstance(structured, dict) and structured.get("status") == "rejected":
        raise HTTPException(
            status_code=422,
            detail=structured.get("reason", "stage_1_extraction_rejected"),
        )

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
        "has_incident_image": has_incident_image,
    }


@app.post("/generate-visualization")
async def generate_visualization(
    analysis_data: str = Form(...),
    selected_scenario: str = Form(...),
    incident_image: UploadFile | None = File(None),
    model: str | None = Form(None),
) -> dict:
    try:
        payload = json.loads(analysis_data)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="analysis_data must be valid JSON")

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="analysis_data must be a JSON object")

    structured = payload.get("structured")
    temporal = payload.get("temporal", [])
    factors = payload.get("situational_factors", [])
    constraints = payload.get("operational_constraints", [])
    scenarios = payload.get("scenarios", [])

    if not isinstance(scenarios, list):
        raise HTTPException(status_code=400, detail="scenarios must be a list")

    scenario_names = [s.get("scenario") for s in scenarios if isinstance(s, dict)]
    if selected_scenario not in scenario_names:
        raise HTTPException(status_code=400, detail="Invalid scenario selection")

    prompt_bundle = build_visualization_request_from_analyze(
        {
            "situational_factors": factors,
            "operational_constraints": constraints,
            "temporal": temporal,
            "scenarios": scenarios,
        },
        scenario=selected_scenario,
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

    incident_pil = None
    if incident_image is not None:
        incident_pil = await validate_and_convert_image(incident_image)

    requested_model = model or "gemini-3-pro-image-preview"
    resolved_model, family = normalize_image_model(requested_model)

    gemini_images = call_image_generation(
        model=resolved_model,
        family=family,
        prompt=prompt_bundle["prompt"],
        context_note=spatial.get("context_note", ""),
        image_urls=image_urls,
        incident_image=incident_pil,
    )

    if incident_pil is not None:
        incident_pil.close()

    return {
        "stage": "stage_3_scene_simulation",
        "selected_scenario": selected_scenario,
        "prompt": prompt_bundle["prompt"],
        "overlay_label": prompt_bundle["overlay_label"],
        "spatial_context": spatial,
        "image_model_requested": requested_model,
        "image_model_used": resolved_model,
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
