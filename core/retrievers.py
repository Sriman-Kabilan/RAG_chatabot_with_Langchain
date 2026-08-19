"""Retriever factories: vectorstore-backed, contextual compression, and Cohere rerank."""

from __future__ import annotations

from typing import Optional

from langchain_cohere import CohereRerank
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter,
    LongContextReorder,
)
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

from core.config import (
    COHERE_RERANK_MODEL,
    DEFAULT_BASE_RETRIEVER_K,
    DEFAULT_COHERE_TOP_N,
    DEFAULT_COMPRESSION_K,
)

# Type alias used across the module.
Retriever = BaseRetriever


def create_vectorstore_backed_retriever(
    vector_store: Chroma,
    search_type: str = "similarity",
    k: int = DEFAULT_BASE_RETRIEVER_K,
    score_threshold: Optional[float] = None,
) -> Retriever:
    """Create a plain vectorstore-backed retriever.

    Args:
        vector_store: A Chroma vector store.
        search_type: ``"similarity"``, ``"mmr"`` or ``"similarity_score_threshold"``.
        k: Number of documents to return.
        score_threshold: Minimum relevance threshold (only used with
            ``similarity_score_threshold``).

    Returns:
        A LangChain ``BaseRetriever``.
    """
    search_kwargs: dict = {"k": k}
    if score_threshold is not None:
        search_kwargs["score_threshold"] = score_threshold

    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )


def create_compression_retriever(
    embeddings: Embeddings,
    base_retriever: Retriever,
    chunk_size: int = 500,
    k: int = DEFAULT_COMPRESSION_K,
    similarity_threshold: Optional[float] = None,
) -> Retriever:
    """Build a contextual-compression retriever.

    The base retriever is wrapped with a ``DocumentCompressorPipeline`` that:
      1. splits each document into smaller chunks,
      2. removes redundant chunks (``EmbeddingsRedundantFilter``),
      3. keeps the ``k`` most relevant chunks (``EmbeddingsFilter``),
      4. reorders the chunks so the most relevant are at the edges
         (``LongContextReorder``).

    Args:
        embeddings: Embeddings used by the filters.
        base_retriever: The vectorstore-backed retriever.
        chunk_size: Splitting size used by the pipeline.
        k: Number of relevant chunks to keep.
        similarity_threshold: Minimum relevance threshold (default: None).

    Returns:
        A ``ContextualCompressionRetriever``.
    """
    # 1. split documents into smaller chunks
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0, separator=". ")

    # 2. remove redundant documents
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)

    # 3. keep documents most relevant to the query
    relevant_filter = EmbeddingsFilter(
        embeddings=embeddings,
        k=k,
        similarity_threshold=similarity_threshold,
    )

    # 4. reorder so less relevant documents sit in the middle
    reordering = LongContextReorder()

    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter, reordering]
    )

    return ContextualCompressionRetriever(
        base_compressor=pipeline_compressor,
        base_retriever=base_retriever,
    )


def create_cohere_rerank_retriever(
    base_retriever: Retriever,
    cohere_api_key: str,
    model: str = COHERE_RERANK_MODEL,
    top_n: int = DEFAULT_COHERE_TOP_N,
) -> Retriever:
    """Wrap the base retriever with a Cohere reranker.

    The Cohere rerank endpoint reorders the retrieved results based on
    semantic relevance to the query.

    Args:
        base_retriever: The vectorstore-backed retriever.
        cohere_api_key: Cohere API key.
        model: Cohere rerank model name.
        top_n: Number of results returned after reranking.

    Returns:
        A ``ContextualCompressionRetriever`` using ``CohereRerank``.
    """
    compressor = CohereRerank(
        cohere_api_key=cohere_api_key,
        model=model,
        top_n=top_n,
    )

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )


def create_retriever(
    vector_store: Chroma,
    embeddings: Embeddings,
    retriever_type: str = "Vectorstore backed retriever",
    cohere_api_key: str = "",
    base_retriever_search_type: str = "similarity",
    base_retriever_k: int = DEFAULT_BASE_RETRIEVER_K,
    compression_retriever_k: int = DEFAULT_COMPRESSION_K,
    cohere_model: str = COHERE_RERANK_MODEL,
    cohere_top_n: int = DEFAULT_COHERE_TOP_N,
) -> Retriever:
    """Create a retriever based on the requested type.

    Supported types (see ``list_retriever_types`` in ``core.config``):

    - ``"Vectorstore backed retriever"``: plain semantic search.
    - ``"Contextual compression"``: pipeline of splitting, redundancy
      removal, relevance filtering and reordering.
    - ``"Cohere reranker"``: rerank results via the Cohere API.

    Args:
        vector_store: The Chroma vector store.
        embeddings: The embeddings used by the vector store.
        retriever_type: One of the supported retriever types.
        cohere_api_key: Required when ``retriever_type`` is the Cohere reranker.
        base_retriever_search_type: Base search type (``"similarity"``, ``"mmr"``, ...).
        base_retriever_k: Number of documents fetched by the base retriever.
        compression_retriever_k: ``k`` used by ``EmbeddingsFilter``.
        cohere_model: Cohere rerank model.
        cohere_top_n: Top-n results from Cohere rerank.

    Returns:
        A LangChain ``BaseRetriever``.
    """
    base_retriever = create_vectorstore_backed_retriever(
        vector_store,
        search_type=base_retriever_search_type,
        k=base_retriever_k,
    )

    if retriever_type == "Vectorstore backed retriever":
        return base_retriever

    if retriever_type == "Contextual compression":
        return create_compression_retriever(
            embeddings=embeddings,
            base_retriever=base_retriever,
            k=compression_retriever_k,
        )

    if retriever_type == "Cohere reranker":
        return create_cohere_rerank_retriever(
            base_retriever=base_retriever,
            cohere_api_key=cohere_api_key,
            model=cohere_model,
            top_n=cohere_top_n,
        )

    raise ValueError(f"Unknown retriever type: {retriever_type!r}")