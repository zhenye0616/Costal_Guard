import os


def call_llm(prompt: str, model: str, temperature: float = 0.1) -> str:
    api_key = (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GENAI_API_KEY")
        or os.getenv("GOOGLE_GENAI_API_KEY")
    )
    if not api_key:
        raise RuntimeError("Missing API key (GOOGLE_API_KEY/GENAI_API_KEY)")

    try:
        import google.genai as genai  # type: ignore

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "temperature": temperature,
                "response_mime_type": "application/json",
            },
        )
        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("Empty response from LLM")
        return text
    except ImportError:
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=api_key)
        llm = genai.GenerativeModel(model)
        response = llm.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "response_mime_type": "application/json",
            },
        )
        if not response.text:
            raise RuntimeError("Empty response from LLM")
        return response.text
