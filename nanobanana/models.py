# nanobanana/models.py

GEMINI_IMAGE_MODELS = {
    "gemini-3-pro-image-preview",
    "gemini-3-image-preview",
    "gemini-2.0-flash-image-generation",
}

IMAGEN_MODELS = {
    "imagen-3.0-generate-002",
}

LEGACY_ALIASES = {
    "gemini-pro-image": "gemini-3-pro-image-preview",
}


def normalize_image_model(model: str) -> tuple[str, str]:
    """
    Returns (normalized_model, api_family)
    api_family ∈ {"gemini", "imagen"}
    """
    model = (model or "").strip()

    if model in LEGACY_ALIASES:
        model = LEGACY_ALIASES[model]

    if model in GEMINI_IMAGE_MODELS:
        return model, "gemini"

    if model in IMAGEN_MODELS:
        return model, "imagen"

    raise ValueError(f"Unsupported image model: {model}")
