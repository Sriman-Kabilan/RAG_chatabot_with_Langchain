"""DeepEval evaluation configuration.

Centralises provider settings, vectorstore paths, and test-question
datasets so the evaluation scripts stay clean.

Usage:
    1. Copy ``.env.example`` → ``.env`` and fill in your API keys, OR
    2. Override the constants below directly (not recommended for secrets).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
VECTOR_STORES_DIR: Path = PROJECT_ROOT / "data" / "vector_stores"

# ---------------------------------------------------------------------------
# Provider / model selection for the evaluation run
# ---------------------------------------------------------------------------
# Which LLM provider to use for generating RAG answers during evaluation.
# Choices: "OpenAI", "Google", "HuggingFace"
EVAL_LLM_PROVIDER: str = os.getenv("EVAL_LLM_PROVIDER", "OpenAI")

# Which chat model to use (must match a model in CONFIG_MODELS for the provider).
EVAL_LLM_MODEL: str = os.getenv("EVAL_LLM_MODEL", "gpt-4o-mini")

# LLM parameters
EVAL_TEMPERATURE: float = float(os.getenv("EVAL_TEMPERATURE", "0.3"))
EVAL_TOP_P: float = float(os.getenv("EVAL_TOP_P", "0.95"))

# Which vectorstore to load for evaluation.
EVAL_VECTORSTORE_NAME: str = os.getenv("EVAL_VECTORSTORE_NAME", "RagDB")

# Which retriever type to use:
#   "Vectorstore backed retriever" | "Contextual compression" | "Cohere reranker"
EVAL_RETRIEVER_TYPE: str = os.getenv(
    "EVAL_RETRIEVER_TYPE", "Vectorstore backed retriever"
)

# ---------------------------------------------------------------------------
# API keys (read from environment / .env)
# ---------------------------------------------------------------------------
API_KEYS: dict = {
    "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
    "google_api_key": os.getenv("GOOGLE_API_KEY", ""),
    "hf_api_key": os.getenv("HUGGINGFACEHUB_API_TOKEN", ""),
    "cohere_api_key": os.getenv("COHERE_API_KEY", ""),
}

# ---------------------------------------------------------------------------
# DeepEval metric thresholds (0.0 – 1.0)
# ---------------------------------------------------------------------------
FAITHFULNESS_THRESHOLD: float = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.7"))
ANSWER_RELEVANCY_THRESHOLD: float = float(os.getenv("ANSWER_RELEVANCY_THRESHOLD", "0.7"))
CONTEXTUAL_PRECISION_THRESHOLD: float = float(os.getenv("CONTEXTUAL_PRECISION_THRESHOLD", "0.7"))
CONTEXTUAL_RECALL_THRESHOLD: float = float(os.getenv("CONTEXTUAL_RECALL_THRESHOLD", "0.7"))
CONTEXTUAL_RELEVANCY_THRESHOLD: float = float(os.getenv("CONTEXTUAL_RELEVANCY_THRESHOLD", "0.7"))
HALLUCINATION_THRESHOLD: float = float(os.getenv("HALLUCINATION_THRESHOLD", "0.5"))

# ---------------------------------------------------------------------------
# Test dataset
# ---------------------------------------------------------------------------
# Each entry is a dict with:
#   "input"            (str)  – the user question
#   "expected_output"  (str)  – the ideal/gold answer (used by Contextual Recall)
#
# 👉  CUSTOMIZE these questions to match the documents in your vectorstore.
#     The default questions below are generic placeholders.
TEST_QUESTIONS: list[dict[str, str]] = [
    {
        "input": "What is the main topic of the uploaded document?",
        "expected_output": "The document discusses the main topic covered in the uploaded files.",
    },
    {
        "input": "Can you summarize the key points?",
        "expected_output": "The key points include the major findings and conclusions presented in the document.",
    },
    {
        "input": "What methodology or approach is described?",
        "expected_output": "The document describes the methodology or approach used in the study or project.",
    },
]
