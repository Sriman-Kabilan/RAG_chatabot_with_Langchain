"""Streamlit sidebar and document/vectorstore chooser."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional

import streamlit as st

from core.config import (
    CONFIG_MODELS,
    LOCAL_VECTOR_STORE_DIR,
    dict_welcome_message,
    list_LLM_providers,
    list_retriever_types,
)
from core.embeddings import select_embeddings_model
from core.llm import create_llm
from core.retrievers import create_retriever
from core.chain import create_retrieval_chain
from core.document_loader import (
    delete_temp_files,
    langchain_document_loader,
    save_uploaded_files,
    split_documents_to_chunks,
    ensure_temp_dir,
)
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

# Type alias for the vectorstore creation callback used by buttons.
VectorStoreFactory = Callable[[], None]


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    """Initialize all Streamlit session state variables once."""
    st.session_state.setdefault("openai_api_key", "")
    st.session_state.setdefault("google_api_key", "")
    st.session_state.setdefault("hf_api_key", "")
    st.session_state.setdefault("cohere_api_key", "")
    st.session_state.setdefault("LLM_provider", "OpenAI")
    st.session_state.setdefault("selected_model", CONFIG_MODELS["OpenAI"]["models"][0])
    st.session_state.setdefault("temperature", 0.5)
    st.session_state.setdefault("top_p", 0.95)
    st.session_state.setdefault("retriever_type", list_retriever_types[1])
    st.session_state.setdefault("assistant_language", "english")
    st.session_state.setdefault("max_history_turns", 20)
    st.session_state.setdefault("error_message", "")
    st.session_state.setdefault("load_error", "")
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("messages", [])


def get_api_keys() -> dict:
    """Return the current API keys from the session state as a plain dict."""
    return {
        "openai_api_key": st.session_state.get("openai_api_key", ""),
        "google_api_key": st.session_state.get("google_api_key", ""),
        "hf_api_key": st.session_state.get("hf_api_key", ""),
        "cohere_api_key": st.session_state.get("cohere_api_key", ""),
    }


def has_api_key() -> bool:
    """Return True if the API key of the selected provider has been provided."""
    provider = st.session_state.LLM_provider
    key_field = CONFIG_MODELS[provider]["api_key_field"]
    return bool(st.session_state.get(key_field, ""))


def validate_credentials() -> List[str]:
    """Return a list of missing credentials for the selected provider/retriever."""
    error_messages: List[str] = []
    if not has_api_key():
        error_messages.append(f"insert your {st.session_state.LLM_provider} API key")
    if (
        st.session_state.retriever_type == list_retriever_types[0]
        and not st.session_state.get("cohere_api_key", "")
    ):
        error_messages.append("insert your Cohere API key")
    return error_messages


def list_existing_vectorstores() -> List[str]:
    """Return the names of the vectorstores saved under data/vector_stores."""
    if not LOCAL_VECTOR_STORE_DIR.exists():
        return []
    return sorted(
        p.name for p in LOCAL_VECTOR_STORE_DIR.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


# ---------------------------------------------------------------------------
# Sidebar UI
# ---------------------------------------------------------------------------

def _expander_model_parameters(provider: str) -> None:
    """Add an API key text_input and a streamlit expander for model + parameters."""
    st.session_state.LLM_provider = provider
    cfg = CONFIG_MODELS[provider]
    key_field = cfg["api_key_field"]

    # Clear the API keys of the other providers.
    for field in [c["api_key_field"] for c in CONFIG_MODELS.values()]:
        if field != key_field:
            st.session_state[field] = ""

    st.session_state[key_field] = st.text_input(
        cfg["api_key_label"],  # type: ignore[arg-type]
        type="password",
        placeholder="insert your API key",
    )

    with st.expander("**Models and parameters**"):
        st.session_state.selected_model = st.selectbox(
            f"Choose {provider} model", cfg["models"]  # type: ignore[arg-type]
        )
        st.session_state.temperature = st.slider(
            "temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1
        )
        st.session_state.top_p = st.slider(
            "top_p", min_value=0.0, max_value=1.0, value=0.95, step=0.05
        )


def _render_sidebar() -> None:
    """Render the left-hand sidebar controls."""
    with st.sidebar:
        st.caption(
            "🚀 A retrieval augmented generation chatbot powered by 🔗 Langchain, "
            "Cohere, OpenAI, Google Generative AI and 🤗"
        )
        st.write("")

        llm_chooser = st.radio(
            "Select provider",
            list_LLM_providers,
            captions=[
                "[OpenAI pricing page](https://openai.com/pricing)",
                "Rate limit: 60 requests per minute.",
                "**Free access.**",
            ],
        )

        st.divider()
        provider_map = {
            list_LLM_providers[0]: "OpenAI",
            list_LLM_providers[1]: "Google",
            list_LLM_providers[2]: "HuggingFace",
        }
        _expander_model_parameters(provider_map[llm_chooser])

        # Assistant language
        st.write("")
        st.session_state.assistant_language = st.selectbox(
            "Assistant language", list(dict_welcome_message.keys())
        )

        st.divider()
        st.subheader("Retrievers")
        st.session_state.retriever_type = st.selectbox(
            "Select retriever type", list_retriever_types
        )
        st.write("")
        if st.session_state.retriever_type == list_retriever_types[0]:  # Cohere
            st.session_state.cohere_api_key = st.text_input(
                "Cohere API Key - [Get an API key](https://dashboard.cohere.com/api-keys)",
                type="password",
                placeholder="insert your API key",
            )

        st.divider()
        st.subheader("Memory")
        st.session_state.max_history_turns = st.slider(
            "Max conversation turns kept in memory",
            min_value=1,
            max_value=30,
            value=20,
        )

        st.write("\n\n")
        st.write(
            f"ℹ _Your {st.session_state.LLM_provider} API key, "
            f"'{st.session_state.selected_model}' parameters, and "
            f"{st.session_state.retriever_type} are only considered when "
            "loading or creating a vectorstore._"
        )


def _tab_new_vectorstore() -> None:
    """Render the 'Create a new Vectorstore' tab."""
    st.session_state.uploaded_file_list = st.file_uploader(
        label="**Select documents**",
        accept_multiple_files=True,
        type=["pdf", "txt", "docx", "csv"],
    )
    st.session_state.vector_store_name = st.text_input(
        label="**Documents will be loaded, embedded and ingested into a "
        "vectorstore (Chroma dB). Please provide a valid dB name.**",
        placeholder="Vectorstore name",
    )
    st.button("Create Vectorstore", on_click=create_vectorstore_from_ui)
    if st.session_state.error_message:
        st.warning(st.session_state.error_message)


def _tab_open_vectorstore() -> None:
    """Render the 'Open a saved Vectorstore' tab."""
    existing_stores = list_existing_vectorstores()
    if not existing_stores:
        st.info("No vectorstores found. Create one in the other tab first.")
    else:
        st.session_state.selected_vectorstore_name = st.selectbox(
            "Select a saved Vectorstore", existing_stores
        )
        st.button("Load Vectorstore", on_click=load_saved_vectorstore)
    if st.session_state.get("load_error"):
        st.warning(st.session_state.load_error)


def sidebar_and_document_chooser() -> None:
    """Main entry point: render the sidebar, the tabs, and handle actions."""
    _render_sidebar()

    # Tabbed Pane: Create a new Vectorstore | Open a saved Vectorstore
    tab_new_vectorstore, tab_open_vectorstore = st.tabs(
        ["Create a new Vectorstore", "Open a saved Vectorstore"]
    )
    with tab_new_vectorstore:
        _tab_new_vectorstore()
    with tab_open_vectorstore:
        _tab_open_vectorstore()


# ---------------------------------------------------------------------------
# Vectorstore / chain actions
# ---------------------------------------------------------------------------

def clear_chat_history() -> None:
    """Clear chat history and memory."""
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": dict_welcome_message[st.session_state.assistant_language],
        }
    ]
    st.session_state.chat_history = []


def build_rag_system(vector_store: Chroma, embeddings, name: str = "") -> None:
    """Create the retriever and the chain, and reset the chat."""
    retriever = create_retriever(
        vector_store=vector_store,
        embeddings=embeddings,
        retriever_type=st.session_state.retriever_type,
        cohere_api_key=st.session_state.cohere_api_key,
    )
    llm = create_llm(
        provider=st.session_state.LLM_provider,
        model=st.session_state.selected_model,
        api_keys=get_api_keys(),
        temperature=st.session_state.temperature,
        top_p=st.session_state.top_p,
    )
    chain = create_retrieval_chain(
        retriever=retriever,
        llm=llm,
        language=st.session_state.assistant_language,
    )

    st.session_state.vector_store = vector_store
    st.session_state.retriever = retriever
    st.session_state.chain = chain

    clear_chat_history()
    if name:
        st.success(f"Ready to chat with **{name}**.")


def create_vectorstore_from_ui() -> None:
    """Process the uploaded documents and build the RAG system."""
    error_messages = validate_credentials()
    if not st.session_state.get("uploaded_file_list"):
        error_messages.append("select documents to upload")
    if not st.session_state.get("vector_store_name", "").strip():
        error_messages.append("provide a Vectorstore name")

    if error_messages:
        st.session_state.error_message = "Please " + ", ".join(error_messages) + "."
        return
    st.session_state.error_message = ""

    with st.spinner("Creating vectorstore..."):
        try:
            # 1. Prepare the temp upload directory
            ensure_temp_dir()
            delete_temp_files()

            # 2. Upload selected documents to the temp directory
            save_uploaded_files(st.session_state.uploaded_file_list)

            # 3. Load documents with LangChain loaders
            documents = langchain_document_loader()
            if not documents:
                st.warning("No documents could be loaded.")
                return

            # 4. Split documents to chunks
            chunks = split_documents_to_chunks(documents)

            # 5. Embeddings
            embeddings = select_embeddings_model(
                provider=st.session_state.LLM_provider,
                api_keys=get_api_keys(),
            )

            # 6. Create and persist the Chroma vectorstore
            store_name = st.session_state.vector_store_name.strip()
            persist_directory = LOCAL_VECTOR_STORE_DIR / store_name
            vector_store = Chroma(
                collection_name="langchain",
                persist_directory=str(persist_directory),
                embedding_function=embeddings,
            )
            vector_store.add_documents(chunks)
            st.info(f"Vectorstore **{store_name}** is created successfully.")

            # 7. Create retriever + chain
            build_rag_system(vector_store, embeddings, store_name)

        except Exception as error:
            st.error(f"{type(error).__name__}: {error}")
            logger.exception("Failed to create the vectorstore")


def load_saved_vectorstore() -> None:
    """Load a saved Chroma vectorstore and build the RAG system."""
    error_messages = validate_credentials()
    if error_messages:
        st.session_state.load_error = "Please " + ", ".join(error_messages) + "."
        return
    st.session_state.load_error = ""

    name = st.session_state.get("selected_vectorstore_name")
    if not name:
        st.session_state.load_error = "Please select a vectorstore."
        return

    with st.spinner("Loading vectorstore..."):
        try:
            embeddings = select_embeddings_model(
                provider=st.session_state.LLM_provider,
                api_keys=get_api_keys(),
            )
            vector_store = Chroma(
                collection_name="langchain",
                persist_directory=str(LOCAL_VECTOR_STORE_DIR / name),
                embedding_function=embeddings,
            )
            build_rag_system(vector_store, embeddings, name)
            st.info(f"**{name}** is loaded successfully.")
        except Exception as error:
            st.error(f"{type(error).__name__}: {error}")
            logger.exception("Failed to load the vectorstore")