import base64
import datetime as dt
import json
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import HTTPException
from google import genai
from google.genai import types
from PIL import Image
import requests

from schema_extraction.constants import CASE_TYPES

BASE_DIR = Path(__file__).resolve().parent
INCIDENT_ROOT = (BASE_DIR / "incident_context").resolve()

load_dotenv()

logger = logging.getLogger(__name__)


def _maps_key() -> str | None:
    key = (
        os.getenv("GOOGLE_MAPS_API_KEY")
        or os.getenv("MAPS_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    if not key or key == "YOUR_KEY":
        return None
    return key


def geocode_location(query: str) -> Tuple[float, float] | None:
    if not query:
        return None
    api_key = _maps_key()
    if not api_key:
        logger.info("Geocoding skipped; missing maps API key")
        return None
    try:
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"address": query, "key": api_key},
            timeout=10,
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") != "OK":
            logger.warning(
                "Geocoding failed",
                extra={"status": payload.get("status"), "query": query},
            )
            return None
        results = payload.get("results", [])
        if not results:
            return None
        loc = results[0].get("geometry", {}).get("location", {})
        lat = loc.get("lat")
        lon = loc.get("lng")
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            return float(lat), float(lon)
    except Exception:
        logger.exception("Geocoding request failed", extra={"query": query})
    return None


def _safe_write_text(path: Path, content: str) -> None:
    try:
        path.write_text(content, encoding="utf-8")
    except Exception:
        logger.exception("Failed to write debug text file", extra={"path": str(path)})


def _safe_write_bytes(path: Path, content: bytes) -> None:
    try:
        path.write_bytes(content)
    except Exception:
        logger.exception("Failed to write debug bytes file", extra={"path": str(path)})


def _infer_ext(content_type: str, fallback: str = ".bin") -> str:
    if not content_type:
        return fallback
    ct = content_type.split(";")[0].strip().lower()
    if ct == "image/jpeg":
        return ".jpg"
    if ct == "image/png":
        return ".png"
    if ct == "image/webp":
        return ".webp"
    if ct == "image/gif":
        return ".gif"
    return fallback


def _create_debug_dir() -> Path:
    root = Path(os.getenv("IMAGE_DEBUG_DIR", BASE_DIR / "image_generation_debug"))
    run_id = f"{dt.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
    debug_dir = root / run_id
    debug_dir.mkdir(parents=True, exist_ok=False)
    return debug_dir


def load_incident_json(rel_path: str) -> Tuple[Any, str]:
    target = (BASE_DIR / rel_path).resolve()
    if INCIDENT_ROOT not in target.parents and target != INCIDENT_ROOT:
        raise HTTPException(status_code=400, detail="source_path must be under incident_context/")
    if not target.exists():
        raise HTTPException(status_code=404, detail="source_path not found")
    if target.suffix.lower() != ".txt":
        raise HTTPException(status_code=400, detail="Only .txt source_path is supported")
    with open(target, "r", encoding="utf-8") as fh:
        return fh.read(), "txt"


def load_cached_structured(source_path: str) -> Any:
    try:
        incident_dir = Path(source_path).parent.name
        candidate = BASE_DIR / "structured_outputs" / f"{incident_dir}.json"
        if not candidate.exists():
            return None

        with open(candidate, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        ic = data.get("incident_classification", {})
        ct = ic.get("case_type")
        if ct not in CASE_TYPES:
            ic["case_type"] = (
                "suspicious_vessel_activity"
                if ct == "suspicious_activity"
                else "unknown"
            )
            data["incident_classification"] = ic

        return data
    except Exception:
        return None


def extract_lat_lon(structured: Any) -> Tuple[float, float]:
    lat = lon = None

    if hasattr(structured, "location_context"):
        loc = structured.location_context
        lat = getattr(loc, "lat", None)
        lon = getattr(loc, "lon", None)

    if lat is None or lon is None:
        loc = structured.get("location_context") if isinstance(structured, dict) else None
        if isinstance(loc, dict):
            lat = loc.get("lat")
            lon = loc.get("lon")

    if lat is None or lon is None:
        raise HTTPException(status_code=400, detail="lat/lon required for spatial context")

    return lat, lon


def call_image_generation(
    *,
    model: str,
    family: str,
    prompt: str,
    context_note: str,
    image_urls: List[str],
    incident_image: Image.Image | None = None,
) -> List[dict]:
    """
    Official Gemini image generation per:
    https://ai.google.dev/gemini-api/docs/image-generation#model-selection
    """

    logger.info(
        "Image generation requested",
        extra={"model": model, "family": family, "image_url_count": len(image_urls)},
    )

    debug_dir = _create_debug_dir()
    logger.info("Image generation debug dir created", extra={"path": str(debug_dir)})

    api_key = (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GENAI_API_KEY")
        or os.getenv("GOOGLE_GENAI_API_KEY")
    )
    if not api_key:
        logger.error("Missing Gemini API key")
        raise HTTPException(status_code=500, detail="Missing Gemini API key")

    combined_prompt = "\n".join(
        [
            prompt,
            "",
            context_note,
        ]
    )
    logger.info(
        "Prepared combined prompt",
        extra={
            "prompt_chars": len(combined_prompt),
            "context_note_chars": len(context_note or ""),
        },
    )

    client = genai.Client(api_key=api_key)
    logger.info("Initialized Gemini client")

    # ------------------------------------------------------------------
    # Gemini multimodal image generation -> generate_content
    # ------------------------------------------------------------------
    if family == "gemini":
        aspect_ratio = os.getenv("GEMINI_IMAGE_ASPECT_RATIO", "1:1")
        image_size = os.getenv("GEMINI_IMAGE_SIZE", "1024x1024")
        logger.info(
            "Gemini image config",
            extra={"aspect_ratio": aspect_ratio, "image_size": image_size},
        )

        request_snapshot: Dict[str, Any] = {
            "timestamp_utc": dt.datetime.utcnow().isoformat() + "Z",
            "model": model,
            "family": family,
            "aspect_ratio": aspect_ratio,
            "image_size": image_size,
            "response_modalities": ["IMAGE"],
            "prompt": prompt,
            "context_note": context_note,
            "combined_prompt": combined_prompt,
            "image_urls": image_urls,
        }
        _safe_write_text(debug_dir / "request.json", json.dumps(request_snapshot, indent=2, ensure_ascii=True))
        _safe_write_text(debug_dir / "prompt.txt", prompt)
        _safe_write_text(debug_dir / "context_note.txt", context_note)
        _safe_write_text(debug_dir / "combined_prompt.txt", combined_prompt)

        image_inputs: List[Image.Image] = []
        image_meta: List[Dict[str, Any]] = []
        if incident_image is not None:
            if incident_image.mode != "RGB":
                incident_image = incident_image.convert("RGB")
            incident_path = debug_dir / "input_incident.png"
            try:
                incident_image.save(incident_path, format="PNG")
            except Exception:
                logger.exception("Failed to write incident debug image")
            image_inputs.append(incident_image.copy())
            image_meta.append(
                {
                    "source": "incident_upload",
                    "type": "spatial_ground_truth",
                    "file": incident_path.name,
                    "mode": incident_image.mode,
                    "size": incident_image.size,
                }
            )
        if image_urls:
            logger.info("Fetching context images", extra={"image_url_count": len(image_urls)})
            map_index = 0
            for url in image_urls:
                try:
                    resp = requests.get(url, timeout=10)
                    resp.raise_for_status()
                    content_type = resp.headers.get("content-type", "")
                    raw_ext = _infer_ext(content_type)
                    raw_path = debug_dir / f"input_{map_index}_raw{raw_ext}"
                    _safe_write_bytes(raw_path, resp.content)

                    img = Image.open(BytesIO(resp.content))
                    if img.mode not in ("RGB", "L"):
                        img = img.convert("RGB")
                    image_inputs.append(img.copy())
                    converted_path = debug_dir / f"input_{map_index}_converted.png"
                    img.save(converted_path, format="PNG")
                    image_meta.append(
                        {
                            "source": "google_maps",
                            "index": map_index,
                            "url": url,
                            "status_code": resp.status_code,
                            "content_type": content_type,
                            "raw_file": raw_path.name,
                            "converted_file": converted_path.name,
                            "mode": img.mode,
                            "size": img.size,
                            "format": img.format,
                        }
                    )
                    img.close()
                    map_index += 1
                except Exception:
                    logger.exception("Failed to fetch context image", extra={"url": url})
                    raise HTTPException(status_code=502, detail="Failed to fetch context imagery")
        _safe_write_text(debug_dir / "input_images.json", json.dumps(image_meta, indent=2, ensure_ascii=True))

        logger.info("Calling Gemini generate_content")
        try:
            response = client.models.generate_content(
                model=model,
                contents=[combined_prompt, *image_inputs],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio,
                        image_size=image_size,
                    ),
                ),
            )
        except Exception:
            logger.exception("Gemini generate_content failed")
            raise

        images: List[dict] = []
        candidate_count = len(getattr(response, "candidates", []) or [])
        logger.info("Gemini generate_content response received", extra={"candidate_count": candidate_count})
        _safe_write_text(
            debug_dir / "response_meta.json",
            json.dumps({"candidate_count": candidate_count}, indent=2, ensure_ascii=True),
        )
        if candidate_count > 0:
            parts = response.candidates[0].content.parts
            logger.info("Parsing Gemini response parts", extra={"part_count": len(parts)})
            for idx, part in enumerate(parts):
                if hasattr(part, "inline_data"):
                    data = part.inline_data.data
                    mime_type = part.inline_data.mime_type
                    if isinstance(data, (bytes, bytearray)):
                        data = base64.b64encode(data).decode("ascii")
                    output_b64_path = debug_dir / f"output_{idx}.b64"
                    _safe_write_text(output_b64_path, data)
                    if mime_type:
                        out_ext = _infer_ext(mime_type, fallback=".bin")
                        try:
                            out_bytes = base64.b64decode(data)
                            _safe_write_bytes(debug_dir / f"output_{idx}{out_ext}", out_bytes)
                        except Exception:
                            logger.exception("Failed to decode output image", extra={"index": idx})
                    images.append(
                        {
                            "index": idx,
                            "image_base64": data,
                            "mime_type": mime_type,
                        }
                    )
        logger.info("Gemini images extracted", extra={"image_count": len(images)})
        return images

    # ------------------------------------------------------------------
    # Imagen (image-only) -> generate_images (future-safe)
    # ------------------------------------------------------------------
    if family == "imagen":
        logger.info("Calling Imagen generate_images")
        try:
            response = client.models.generate_images(
                model=model,
                prompt=combined_prompt,
            )
        except Exception:
            logger.exception("Imagen generate_images failed")
            raise
        images = [
            {
                "index": idx,
                "image_base64": img.image_base64,
                "mime_type": img.mime_type,
            }
            for idx, img in enumerate(response.generated_images)
        ]
        logger.info("Imagen images extracted", extra={"image_count": len(images)})
        return images

    logger.error("Unhandled model family", extra={"family": family})
    raise HTTPException(status_code=500, detail=f"Unhandled model family: {family}")
