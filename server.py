import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from schema_extraction.pipeline import extract_structured
from incident_analysis.temporal import derive_temporal
from incident_analysis.situational import derive_factors
from incident_analysis.operational_constraints import derive_constraints
from incident_analysis.scenarios import derive_scenarios

BASE_DIR = Path(__file__).resolve().parent
INCIDENT_ROOT = (BASE_DIR / "incident_context").resolve()


class ExtractRequest(BaseModel):
    source_path: str


class RawRequest(BaseModel):
    source_path: str


app = FastAPI(title="USCG Incident Extraction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables from .env if present
load_dotenv()


def load_incident_json(rel_path: str) -> Any:
    target = (BASE_DIR / rel_path).resolve()
    if INCIDENT_ROOT not in target.parents and target != INCIDENT_ROOT:
        raise HTTPException(status_code=400, detail="source_path must be under incident_context/")
    if not target.exists():
        raise HTTPException(status_code=404, detail="source_path not found")
    if target.suffix.lower() != ".txt":
        raise HTTPException(status_code=400, detail="Only .txt source_path is supported")
    with open(target, "r", encoding="utf-8") as fh:
        base_text = fh.read()
    # If a sibling incident.json exists, append coordinates to aid extraction
    sibling_json = target.parent / "incident.json"
    if sibling_json.exists():
        try:
            with open(sibling_json, "r", encoding="utf-8") as jh:
                loose = json.load(jh)
            lat = loose.get("lat")
            lon = loose.get("lon")
            loc_note = loose.get("location_note") or ""
            if lat is not None and lon is not None:
                base_text += f"\n\nCoordinates:\nLatitude: {lat}\nLongitude: {lon}\n{loc_note}"
        except Exception:
            pass
    return base_text, "txt"


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
    return {"source_path": req.source_path, "result": result}


@app.post("/analyze")
def analyze(req: ExtractRequest) -> Any:
    """
    Run Stage 1 extraction and Stage 2 analysis (temporal, factors, constraints, scenarios).
    """
    payload, fmt = load_incident_json(req.source_path)
    structured = extract_structured(payload, source_format=fmt)
    if isinstance(structured, dict) and structured.get("status") == "rejected":
        return {"source_path": req.source_path, "structured": structured}

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
        "source_path": req.source_path,
        "structured": structured,
        "temporal": temporal,
        "situational_factors": factors,
        "operational_constraints": constraints,
        "scenarios": scenarios,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
