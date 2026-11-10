"""DeepEval RAG evaluation – Approach B (explicit test cases).

This script:
  1. Loads an existing Chroma vectorstore from disk.
  2. Builds the same LCEL RAG chain your Streamlit app uses.
  3. Runs each test question through the chain.
  4. Scores every answer with DeepEval metrics (Faithfulness, Answer
     Relevancy, Contextual Precision/Recall/Relevancy, Hallucination).
  5. Prints a detailed results table and saves a JSON report.

Run with:
    python -m tests.test_rag_eval            (from the project root)
  OR
    deepeval test run tests/test_rag_eval.py  (pytest-style, with dashboard)

Prerequisites:
  - ``pip install deepeval``
  - An OPENAI_API_KEY in your ``.env`` (DeepEval uses GPT under the hood
    for metric evaluation, regardless of the RAG provider).
  - A populated vectorstore in ``data/vector_stores/<name>``.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase
from langchain_chroma import Chroma

# ── project imports ──────────────────────────────────────────────────────
from core.chain import create_retrieval_chain
from core.config import COLLECTION_NAME
from core.embeddings import select_embeddings_model
from core.llm import create_llm
from core.retrievers import create_retriever

from tests.eval_config import (
    API_KEYS,
    ANSWER_RELEVANCY_THRESHOLD,
    CONTEXTUAL_PRECISION_THRESHOLD,
    CONTEXTUAL_RECALL_THRESHOLD,
    CONTEXTUAL_RELEVANCY_THRESHOLD,
    EVAL_LLM_MODEL,
    EVAL_LLM_PROVIDER,
    EVAL_RETRIEVER_TYPE,
    EVAL_TEMPERATURE,
    EVAL_TOP_P,
    EVAL_VECTORSTORE_NAME,
    FAITHFULNESS_THRESHOLD,
    HALLUCINATION_THRESHOLD,
    TEST_QUESTIONS,
    VECTOR_STORES_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────

def _build_chain(provider: str, model: str) -> Any:
    """Load the vectorstore, create embeddings, retriever and chain."""
    persist_dir = VECTOR_STORES_DIR / EVAL_VECTORSTORE_NAME
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"Vectorstore '{EVAL_VECTORSTORE_NAME}' not found at {persist_dir}.\n"
            f"Available stores: {[p.name for p in VECTOR_STORES_DIR.iterdir() if p.is_dir()]}"
        )

    embeddings = select_embeddings_model(provider=provider, api_keys=API_KEYS)

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )

    retriever = create_retriever(
        vector_store=vector_store,
        embeddings=embeddings,
        retriever_type=EVAL_RETRIEVER_TYPE,
        cohere_api_key=API_KEYS.get("cohere_api_key", ""),
    )

    llm = create_llm(
        provider=provider,
        model=model,
        api_keys=API_KEYS,
        temperature=EVAL_TEMPERATURE,
        top_p=EVAL_TOP_P,
    )

    chain = create_retrieval_chain(retriever=retriever, llm=llm)
    return chain


def _run_chain(chain: Any, question: str) -> Dict[str, Any]:
    """Invoke the RAG chain and return answer + retrieval context."""
    result = chain.invoke({"input": question, "chat_history": []})
    answer: str = result.get("answer", "")
    context_docs = result.get("context", [])
    retrieval_context: List[str] = [doc.page_content for doc in context_docs]
    return {"answer": answer, "retrieval_context": retrieval_context}


def build_test_cases(chain: Any) -> List[LLMTestCase]:
    """Run every test question through the RAG chain and build DeepEval test cases."""
    test_cases: List[LLMTestCase] = []

    for i, q in enumerate(TEST_QUESTIONS, start=1):
        question = q["input"]
        expected = q.get("expected_output")
        logger.info("(%d/%d) Querying: %s", i, len(TEST_QUESTIONS), question)

        result = _run_chain(chain, question)

        logger.info("  → Answer:  %s", result["answer"][:120] + "...")
        logger.info("  → Contexts retrieved: %d", len(result["retrieval_context"]))

        tc = LLMTestCase(
            input=question,
            actual_output=result["answer"],
            expected_output=expected,
            retrieval_context=result["retrieval_context"],
            # context is used by HallucinationMetric (list of factual strings)
            context=result["retrieval_context"],
        )
        test_cases.append(tc)

    return test_cases


def get_metrics() -> list:
    """Instantiate and return all DeepEval metrics."""
    return [
        FaithfulnessMetric(threshold=FAITHFULNESS_THRESHOLD),
        AnswerRelevancyMetric(threshold=ANSWER_RELEVANCY_THRESHOLD),
        ContextualPrecisionMetric(threshold=CONTEXTUAL_PRECISION_THRESHOLD),
        ContextualRecallMetric(threshold=CONTEXTUAL_RECALL_THRESHOLD),
        ContextualRelevancyMetric(threshold=CONTEXTUAL_RELEVANCY_THRESHOLD),
        HallucinationMetric(threshold=HALLUCINATION_THRESHOLD),
    ]


def save_report(test_cases: List[LLMTestCase], results: Any) -> Path:
    """Save a JSON evaluation report to the project root."""
    report_dir = Path(__file__).resolve().parent.parent / "data" / "eval_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"eval_report_{timestamp}.json"

    report_data = {
        "timestamp": timestamp,
        "provider": EVAL_LLM_PROVIDER,
        "model": EVAL_LLM_MODEL,
        "vectorstore": EVAL_VECTORSTORE_NAME,
        "retriever_type": EVAL_RETRIEVER_TYPE,
        "num_questions": len(test_cases),
        "test_cases": [],
    }

    for tc in test_cases:
        report_data["test_cases"].append(
            {
                "input": tc.input,
                "actual_output": tc.actual_output,
                "expected_output": tc.expected_output,
                "retrieval_context_count": len(tc.retrieval_context) if tc.retrieval_context else 0,
            }
        )

    report_path.write_text(json.dumps(report_data, indent=2, ensure_ascii=False))
    logger.info("Report saved to %s", report_path)
    return report_path


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point: build chain → run questions → evaluate → save report."""
    logger.info("=" * 60)
    logger.info("DeepEval RAG Evaluation — Approach B")
    logger.info("=" * 60)
    logger.info("Provider     : %s", EVAL_LLM_PROVIDER)
    logger.info("Model        : %s", EVAL_LLM_MODEL)
    logger.info("Vectorstore  : %s", EVAL_VECTORSTORE_NAME)
    logger.info("Retriever    : %s", EVAL_RETRIEVER_TYPE)
    logger.info("Questions    : %d", len(TEST_QUESTIONS))
    logger.info("-" * 60)

    # 1. Build the RAG chain (reuses your existing core modules)
    chain = _build_chain(EVAL_LLM_PROVIDER, EVAL_LLM_MODEL)
    logger.info("RAG chain built successfully ✓")

    # 2. Generate answers for each test question
    test_cases = build_test_cases(chain)
    logger.info("All test cases generated ✓")

    # 3. Define metrics
    metrics = get_metrics()
    logger.info("Metrics: %s", [m.__class__.__name__ for m in metrics])

    # 4. Evaluate with DeepEval
    logger.info("-" * 60)
    logger.info("Running DeepEval evaluation...")
    results = evaluate(test_cases, metrics=metrics)

    # 5. Save report
    save_report(test_cases, results)

    logger.info("=" * 60)
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
