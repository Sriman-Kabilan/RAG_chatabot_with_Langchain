"""Embeddings factories for the supported LLM providers."""

from __future__ import annotations

from typing import List

from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_openai import OpenAIEmbeddings

from core.config import (
    CONFIG_MODELS,
    E5_PASSAGE_PREFIX,
    E5_QUERY_PREFIX,
)


class E5EndpointEmbeddings(HuggingFaceEndpointEmbeddings):
    """Hugging Face Inference API embeddings with E5 query/passage prefixes.

    The intfloat/multilingual-e5 models expect ``query:`` / ``passage:``
    prefixes on the input text to produce well-formed embeddings.
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = [E5_PASSAGE_PREFIX + text for text in texts]
        return super().embed_documents(prefixed)

    def embed_query(self, text: str) -> List[float]:
        return super().embed_query(E5_QUERY_PREFIX + text)


def select_embeddings_model(provider: str, api_keys: dict) -> Embeddings:
    """Instantiate the embeddings model for the selected provider.

    Args:
        provider: One of ``"OpenAI"``, ``"Google"`` or ``"HuggingFace"``.
        api_keys: Mapping with ``openai_api_key``, ``google_api_key``
            and/or ``hf_api_key``.

    Returns:
        A LangChain ``Embeddings`` instance.
    """
    cfg = CONFIG_MODELS[provider]

    if provider == "OpenAI":
        return OpenAIEmbeddings(
            model=cfg["embeddings_model"],  # type: ignore[arg-type]
            api_key=api_keys.get("openai_api_key"),
        )

    if provider == "Google":
        return GoogleGenerativeAIEmbeddings(
            model=cfg["embeddings_model"],  # type: ignore[arg-type]
            google_api_key=api_keys.get("google_api_key"),
        )

    # Hugging Face serverless Inference API (no local download).
    return E5EndpointEmbeddings(
        model=cfg["embeddings_model"],  # type: ignore[arg-type]
        task="feature-extraction",
        huggingfacehub_api_token=api_keys.get("hf_api_key"),
    )