"""RAG chain built with pure LangChain LCEL (no langchain-classic chain factories)."""

from __future__ import annotations

from operator import itemgetter
from typing import Any, Dict, List, Sequence

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

CONTEXTUALIZE_Q_SYSTEM_PROMPT = (
    "Given the following conversation and a follow up question, "
    "rephrase the follow up question to be a standalone question, "
    "in its original language.\n"
    "Do NOT answer the question, just rephrase it if needed, "
    "otherwise return it as is."
)


def answer_template(language: str = "english") -> str:
    """Template passed to the LLM along with the retrieved context.

    The answer is grounded in ``{context}`` and returned in the requested
    ``language``.
    """
    return f"""Answer the question at the end, using only the following context (delimited by <context></context>).
Your answer must be in the language at the end.

<context>
{{context}}
</context>

Question: {{input}}

Language: {language}."""


# ---------------------------------------------------------------------------
# Document formatting helpers
# ---------------------------------------------------------------------------

DEFAULT_DOCUMENT_PROMPT = """Source: {source}
{page_content}"""


def format_documents(documents: Sequence[Document]) -> str:
    """Format a list of documents into a single context string."""
    parts: List[str] = []
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        parts.append(
            DEFAULT_DOCUMENT_PROMPT.format(source=source, page_content=doc.page_content)
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Chain construction
# ---------------------------------------------------------------------------


def create_history_aware_retriever_chain(retriever: Any, llm: Any) -> Any:
    """Rephrase a follow-up question into a standalone question, then retrieve.

    This is the pure-LCEL equivalent of the classic
    ``create_history_aware_retriever``.

    Args:
        retriever: A LangChain retriever (vectorstore-backed, compression, ...).
        llm: A LangChain chat model.

    Returns:
        A runnable that accepts ``{"input": str, "chat_history": list}`` and
        returns a dict with ``"context"`` (retrieved documents) and
        ``"standalone_question"`` (the rephrased query).
    """
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    rephrase_chain = (
        {
            "input": itemgetter("input"),
            "chat_history": itemgetter("chat_history"),
        }
        | contextualize_q_prompt
        | llm
        | StrOutputParser()
    )

    return (
        RunnablePassthrough.assign(standalone_question=rephrase_chain)
        | RunnablePassthrough.assign(
            context=itemgetter("standalone_question") | retriever
        )
    )


def create_stuff_documents_chain_llm(llm: Any, language: str = "english") -> Any:
    """Create the question-answer step of the RAG chain.

    This is the pure-LCEL equivalent of ``create_stuff_documents_chain``.

    Args:
        llm: A LangChain chat model.
        language: The assistant language.

    Returns:
        A runnable that accepts
        ``{"input": str, "chat_history": list, "context": list[Document]}``
        and streams the answer as a string.
    """
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", answer_template(language=language)),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    return (
        {
            "input": itemgetter("input"),
            "chat_history": itemgetter("chat_history"),
            "context": RunnableLambda(lambda x: format_documents(x["context"])),
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )


def create_retrieval_chain(retriever: Any, llm: Any, language: str = "english") -> Any:
    """Compose the full RAG chain.

    Pipeline:
        1. Rephrase the follow-up question into a standalone question.
        2. Retrieve relevant documents for the standalone question.
        3. Answer using the retrieved context + chat history.

    The chain streams the answer token-by-token and also exposes the
    retrieved ``context`` so the UI can display source documents.

    Args:
        retriever: A LangChain retriever.
        llm: A LangChain chat model.
        language: Assistant response language.

    Returns:
        A composed LCEL runnable. When streamed, it yields
        ``{"answer": ...}`` chunks and a final ``{"context": [...]}`` chunk.
    """
    history_aware = create_history_aware_retriever_chain(retriever, llm)
    answer_step = create_stuff_documents_chain_llm(llm, language=language)

    return (
        history_aware
        | RunnableParallel(
            answer=answer_step,
            context=itemgetter("context"),
        )
    )


def build_rag_system_chain(
    retriever: Any,
    llm: Any,
    language: str = "english",
) -> Dict[str, Any]:
    """Build the retriever + chain and return them together.

    Convenience wrapper used by the UI layer.

    Returns:
        ``{"retriever": ..., "chain": ...}``
    """
    chain = create_retrieval_chain(retriever, llm, language=language)
    return {
        "retriever": retriever,
        "chain": chain,
    }