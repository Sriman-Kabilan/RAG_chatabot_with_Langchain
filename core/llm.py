"""LLM factory: instantiate the chat model for the selected provider."""

from __future__ import annotations

from typing import Any, Dict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI

from core.config import CONFIG_MODELS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P


def create_llm(
    provider: str,
    model: str,
    api_keys: Dict[str, str],
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
) -> Any:
    """Instantiate the chat model for the given provider.

    Args:
        provider: ``"OpenAI"``, ``"Google"`` or ``"HuggingFace"``.
        model: The model name (see ``CONFIG_MODELS[provider]["models"]``).
        api_keys: Mapping containing the relevant provider API key.
        temperature: Controls randomness (0.0 - 1.0).
        top_p: Nucleus sampling cutoff (0.0 - 1.0).

    Returns:
        A LangChain chat model instance.
    """
    if provider == "OpenAI":
        return ChatOpenAI(
            model=model,
            api_key=api_keys.get("openai_api_key"),
            temperature=temperature,
            model_kwargs={"top_p": top_p},
        )

    if provider == "Google":
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_keys.get("google_api_key"),
            temperature=temperature,
            top_p=top_p,
        )

    # Hugging Face (chat-completions Inference API; only models served by the
    # router are usable, e.g. Qwen/Qwen2.5-7B-Instruct).
    return ChatHuggingFace(
        llm=HuggingFaceEndpoint(
            repo_id=model,
            huggingfacehub_api_token=api_keys.get("hf_api_key"),
            task="text-generation",
        ),
        temperature=temperature,
        model_kwargs={"top_p": top_p},
        max_tokens=1024,
    )


def llm_parameters(provider: str) -> Dict[str, Any]:
    """Return the display config for a provider (models + default params)."""
    cfg = CONFIG_MODELS[provider]
    return {
        "models": cfg["models"],
        "api_key_field": cfg["api_key_field"],
        "api_key_label": cfg["api_key_label"],
    }