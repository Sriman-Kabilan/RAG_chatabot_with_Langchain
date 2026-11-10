"""Application configuration: providers, models, paths, and constants."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

# Load API keys from a .env file if present (optional).
load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
TMP_DIR: Path = PROJECT_ROOT / "data" / "tmp"
LOCAL_VECTOR_STORE_DIR: Path = PROJECT_ROOT / "data" / "vector_stores"
COLLECTION_NAME: str = "langchain"

# ---------------------------------------------------------------------------
# LLM providers
# ---------------------------------------------------------------------------

# Display labels used in the Streamlit sidebar (in the radio buttons).
list_LLM_providers: List[str] = [
    ":rainbow[**OpenAI**]",
    "**Google Generative AI**",
    ":hugging_face: **HuggingFace**",
]

# provider -> (session_state api key field, chat models, embeddings model, api key label)
CONFIG_MODELS: Dict[str, Dict[str, object]] = {
    "OpenAI": {
        "api_key_field": "openai_api_key",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
        "embeddings_model": "text-embedding-3-small",
        "api_key_label": "OpenAI API Key - [Get an API key](https://platform.openai.com/account/api-keys)",
    },
    "Google": {
        "api_key_field": "google_api_key",
        "models": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
        "embeddings_model": "models/text-embedding-004",
        "api_key_label": "Google API Key - [Get an API key](https://aistudio.google.com/apikey)",
    },
    "HuggingFace": {
        "api_key_field": "hf_api_key",
        "models": [
            "meta-llama/Llama-3.1-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "Qwen/Qwen2.5-72B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "microsoft/phi-4",
        ],
        "embeddings_model": "intfloat/multilingual-e5-large",
        "api_key_label": "HuggingFace API key - [Get an API key](https://huggingface.co/settings/tokens)",
    },
}

# Assistant welcome messages, one per supported language.
dict_welcome_message: Dict[str, str] = {
    "english": "How can I assist you today?",
    "french": "Comment puis-je vous aider aujourd’hui ?",
    "spanish": "¿Cómo puedo ayudarle hoy?",
    "german": "Wie kann ich Ihnen heute helfen?",
    "russian": "Чем я могу помочь вам сегодня?",
    "chinese": "我今天能帮你什么？",
    "arabic": "كيف يمكنني مساعدتك اليوم؟",
    "portuguese": "Como posso ajudá-lo hoje?",
    "italian": "Come posso assistervi oggi?",
    "japanese": "今日はどのようなご用件でしょうか?",
}

# Retriever options shown in the sidebar.
list_retriever_types: List[str] = [
    "Cohere reranker",
    "Contextual compression",
    "Vectorstore backed retriever",
]

# ---------------------------------------------------------------------------
# Default RAG / retrieval parameters
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE: int = 1600
DEFAULT_CHUNK_OVERLAP: int = 200
DEFAULT_BASE_RETRIEVER_K: int = 16
DEFAULT_COMPRESSION_K: int = 20
DEFAULT_COHERE_TOP_N: int = 10
COHERE_RERANK_MODEL: str = "rerank-multilingual-v3.0"

# Embeddings model for the E5 (Hugging Face) endpoint
E5_QUERY_PREFIX: str = "query: "
E5_PASSAGE_PREFIX: str = "passage: "

# Default LLM parameters
DEFAULT_TEMPERATURE: float = 0.5
DEFAULT_TOP_P: float = 0.95

# Supported document extensions
SUPPORTED_EXTENSIONS: List[str] = ["pdf", "txt", "docx", "csv"]