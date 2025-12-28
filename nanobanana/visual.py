"""
Stage 3.1 — Spatial Context Provider (images only, non-inferential).

Produces geographic backdrop imagery metadata (not evidence) using only lat/lon and
optional situational/time hints for camera selection. Enforces wide-angle, cardinal
headings, and rejects forbidden inputs.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

CONTEXT_NOTE = "Images provide geographic backdrop only; not evidence"
OVERLAY_DISCLAIMER = "Hypothetical Visualization — Not Evidence · Not Ground Truth"

# Inputs that must not be present (non-exhaustive but explicit guardrails)
FORBIDDEN_FIELDS = {
    "scenario",
    "scenarios",
    "observations",
    "risk_vectors",
    "primary_risk_vectors",
    "confidence",
    "temporal_measurements",
    "time_since_incident",
    "report_time_local",
    "description",
    "raw_report",
    "raw_text",
    "narrative",
}


def _sanitize_list(values: Sequence[str]) -> List[str]:
    return [v.strip() for v in values if v and isinstance(v, str) and v.strip()]


def _validate_lat_lon(lat: float, lon: float) -> None:
    if not (isinstance(lat, (int, float)) and isinstance(lon, (int, float))):
        raise ValueError("lat and lon must be numeric")
    if math.isnan(lat) or math.isnan(lon) or math.isinf(lat) or math.isinf(lon):
        raise ValueError("lat and lon must be finite")
    if not (-90.0 <= lat <= 90.0):
        raise ValueError("lat must be within [-90, 90]")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError("lon must be within [-180, 180]")


def _maps_key() -> str:
    """
    Prefer a dedicated maps key; fall back to generic Google key if needed.
    """
    return (
        os.getenv("GOOGLE_MAPS_API_KEY")
        or os.getenv("MAPS_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or "YOUR_KEY"
    )


@dataclass
class SpatialContextRequest:
    lat: float
    lon: float
    situational_factors: List[str] = field(default_factory=list)
    time_context: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _validate_lat_lon(self.lat, self.lon)
        self.situational_factors = _sanitize_list(self.situational_factors)
        self.time_context = _sanitize_list(self.time_context)


def _build_street_view_images(lat: float, lon: float) -> List[Dict[str, Any]]:
    headings = [0, 90, 180]  # 2–4 images; cardinal only
    pitch = 0
    fov = 90
    key = _maps_key()
    images: List[Dict[str, Any]] = []
    for heading in headings:
        images.append(
            {
                "type": "street_view",
                "heading": heading,
                "pitch": pitch,
                "fov": fov,
                "url": (
                    "https://maps.googleapis.com/maps/api/streetview"
                    f"?size=640x360&location={lat},{lon}&heading={heading}&pitch={pitch}&fov={fov}&key={key}"
                ),
            }
        )
    return images


def _build_satellite_overview(lat: float, lon: float) -> Dict[str, Any]:
    zoom = 11  # zoomed out; unsuitable for asset identification
    key = _maps_key()
    return {
        "type": "satellite_overview",
        "heading": 0,
        "pitch": 0,
        "fov": 90,
        "zoom": zoom,
        "url": (
            "https://maps.googleapis.com/maps/api/staticmap"
            f"?center={lat},{lon}&zoom={zoom}&size=640x360&maptype=satellite&key={key}"
        ),
    }


def build_spatial_context(req: SpatialContextRequest) -> Dict[str, Any]:
    """
    Produce Stage 3.1 output: wide-angle, cardinal, non-specific imagery metadata.
    """
    images: List[Dict[str, Any]] = []
    images.extend(_build_street_view_images(req.lat, req.lon))
    images.append(_build_satellite_overview(req.lat, req.lon))

    # Trim to max 4 images if future variants add more sources
    images = images[:4]

    return {
        "images": images,
        "context_note": CONTEXT_NOTE,
        "overlay_disclaimer": OVERLAY_DISCLAIMER,
    }


def build_spatial_context_from_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Validate inputs and build spatial context from a mapping containing lat/lon.
    Rejects payloads that include forbidden fields.
    """
    forbidden_present = FORBIDDEN_FIELDS.intersection(set(payload.keys()))
    if forbidden_present:
        raise ValueError(f"forbidden inputs present for Stage 3.1: {sorted(forbidden_present)}")

    lat = payload.get("lat")
    lon = payload.get("lon")
    situational = payload.get("situational_factors") or []
    time_ctx = payload.get("time_context") or payload.get("temporal_context") or []

    req = SpatialContextRequest(lat=lat, lon=lon, situational_factors=situational, time_context=time_ctx)
    return build_spatial_context(req)


__all__ = [
    "SpatialContextRequest",
    "build_spatial_context",
    "build_spatial_context_from_payload",
    "CONTEXT_NOTE",
    "OVERLAY_DISCLAIMER",
    "FORBIDDEN_FIELDS",
]
