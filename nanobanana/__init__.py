"""
Stage 3 — Scene Simulation helpers.
"""

from .prompt import OVERLAY_LABEL, Prompt, build_scene_prompt, build_visualization_request, build_visualization_request_from_analyze
from .visual import (
    CONTEXT_NOTE,
    OVERLAY_DISCLAIMER,
    FORBIDDEN_FIELDS,
    SpatialContextRequest,
    build_spatial_context,
    build_spatial_context_from_payload,
)

__all__ = [
    "Stage3Inputs",
    "OVERLAY_LABEL",
    "build_scene_prompt",
    "build_visualization_request",
    "build_visualization_request_from_analyze",
    "SpatialContextRequest",
    "build_spatial_context",
    "build_spatial_context_from_payload",
    "CONTEXT_NOTE",
    "OVERLAY_DISCLAIMER",
    "FORBIDDEN_FIELDS",
]
