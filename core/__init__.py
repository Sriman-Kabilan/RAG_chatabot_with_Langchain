"""Core RAG building blocks: config, document processing, embeddings, retrievers, LLMs, and chains."""

from core.config import (
    CONFIG_MODELS,
    COLLECTION_NAME,
    LOCAL_VECTOR_STORE_DIR,
    TMP_DIR,
    dict_welcome_message,
    list_LLM_providers,
    list_retriever_types,
)
from core.chain import (
    answer_template,
    build_rag_system_chain,
    create_history_aware_retriever_chain,
    create_retrieval_chain,
    create_stuff_documents_chain_llm,
    format_documents,
)
from core.document_loader import (
    delete_temp_files,
    ensure_temp_dir,
    langchain_document_loader,
    save_uploaded_files,
    split_documents_to_chunks,
)
from core.embeddings import E5EndpointEmbeddings, select_embeddings_model
from core.llm import create_llm, llm_parameters
from core.memory import append_turn, to_message_list, trim_chat_history
from core.retrievers import (
    create_cohere_rerank_retriever,
    create_compression_retriever,
    create_retriever,
    create_vectorstore_backed_retriever,
)

__all__ = [
    # config
    "CONFIG_MODELS",
    "COLLECTION_NAME",
    "LOCAL_VECTOR_STORE_DIR",
    "TMP_DIR",
    "dict_welcome_message",
    "list_LLM_providers",
    "list_retriever_types",
    # chain
    "answer_template",
    "build_rag_system_chain",
    "create_history_aware_retriever_chain",
    "create_retrieval_chain",
    "create_stuff_documents_chain_llm",
    "format_documents",
    # document_loader
    "delete_temp_files",
    "ensure_temp_dir",
    "langchain_document_loader",
    "save_uploaded_files",
    "split_documents_to_chunks",
    # embeddings
    "E5EndpointEmbeddings",
    "select_embeddings_model",
    # llm
    "create_llm",
    "llm_parameters",
    # memory
    "append_turn",
    "to_message_list",
    "trim_chat_history",
    # retrievers
    "create_cohere_rerank_retriever",
    "create_compression_retriever",
    "create_retriever",
    "create_vectorstore_backed_retriever",
]