import os

import ollama

from config import get_ollama_base_url, get_llm_provider, get_nvidia_api_key, get_nvidia_model

_selected_model: str | None = None


def _ollama_client() -> ollama.Client:
    return ollama.Client(host=get_ollama_base_url())


def _nvidia_client():
    from openai import OpenAI
    api_key = get_nvidia_api_key()
    if not api_key:
        api_key = os.environ.get("NVIDIA_API_KEY", "")
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
    )


def list_models() -> list[str]:
    provider = get_llm_provider()
    if provider == "nvidia_nim":
        return [get_nvidia_model() or "meta/llama-3.1-70b-instruct"]
    response = _ollama_client().list()
    return sorted(m.model for m in response.models)


def select_model(model: str) -> None:
    global _selected_model
    _selected_model = model


def get_active_model() -> str | None:
    return _selected_model


def generate_text(prompt: str, model_name: str = None) -> str:
    provider = get_llm_provider()
    model = model_name or _selected_model

    if provider == "nvidia_nim":
        if not model:
            model = get_nvidia_model() or "meta/llama-3.1-70b-instruct"

        client = _nvidia_client()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2048,
        )
        return response.choices[0].message.content.strip()

    # Default: local Ollama
    if not model:
        raise RuntimeError(
            "No Ollama model selected. Call select_model() first or pass model_name."
        )

    response = _ollama_client().chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    return response["message"]["content"].strip()
