"""Pytest-compatible DeepEval tests (for ``deepeval test run``).

Usage:
    deepeval test run tests/test_rag_pytest.py
    deepeval test run tests/test_rag_pytest.py --verbose

This file defines proper ``test_*`` functions that ``deepeval`` and ``pytest``
can discover automatically.  Each test question becomes its own test case
so you get per-question pass/fail granularity.
"""

from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase

from tests.test_rag_eval import _build_chain, _run_chain
from tests.eval_config import (
    ANSWER_RELEVANCY_THRESHOLD,
    CONTEXTUAL_PRECISION_THRESHOLD,
    CONTEXTUAL_RECALL_THRESHOLD,
    CONTEXTUAL_RELEVANCY_THRESHOLD,
    EVAL_LLM_MODEL,
    EVAL_LLM_PROVIDER,
    FAITHFULNESS_THRESHOLD,
    HALLUCINATION_THRESHOLD,
    TEST_QUESTIONS,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rag_chain():
    """Build the RAG chain once per test module."""
    return _build_chain(EVAL_LLM_PROVIDER, EVAL_LLM_MODEL)


@pytest.fixture(scope="module")
def metrics():
    """Instantiate metrics once per test module."""
    return [
        FaithfulnessMetric(threshold=FAITHFULNESS_THRESHOLD),
        AnswerRelevancyMetric(threshold=ANSWER_RELEVANCY_THRESHOLD),
        ContextualPrecisionMetric(threshold=CONTEXTUAL_PRECISION_THRESHOLD),
        ContextualRecallMetric(threshold=CONTEXTUAL_RECALL_THRESHOLD),
        ContextualRelevancyMetric(threshold=CONTEXTUAL_RELEVANCY_THRESHOLD),
        HallucinationMetric(threshold=HALLUCINATION_THRESHOLD),
    ]


# ---------------------------------------------------------------------------
# Parametrized test — one test per question
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "question_data",
    TEST_QUESTIONS,
    ids=[q["input"][:60] for q in TEST_QUESTIONS],
)
def test_rag_answer(rag_chain, metrics, question_data):
    """Evaluate a single RAG question against all metrics."""
    result = _run_chain(rag_chain, question_data["input"])

    test_case = LLMTestCase(
        input=question_data["input"],
        actual_output=result["answer"],
        expected_output=question_data.get("expected_output"),
        retrieval_context=result["retrieval_context"],
        context=result["retrieval_context"],
    )

    assert_test(test_case, metrics)
