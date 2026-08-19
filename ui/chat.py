"""Streamlit chat interface: message display, streaming, and source documents."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import streamlit as st

from core.config import dict_welcome_message
from core.memory import append_turn, trim_chat_history
from ui.sidebar import has_api_key, init_session_state

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source document display
# ---------------------------------------------------------------------------

def display_source_documents(documents: List[Any]) -> None:
    """Display the source documents (name, page and content) in an expander."""
    documents_content = ""
    for document in documents:
        try:
            page = " (Page: " + str(document.metadata["page"]) + ")"
        except Exception:
            page = ""
        documents_content += (
            "**Source: " + str(document.metadata.get("source", "unknown")) + page + "**\n\n"
        )
        documents_content += document.page_content + "\n\n\n"
    st.markdown(documents_content)


# ---------------------------------------------------------------------------
# Response generation
# ---------------------------------------------------------------------------

def get_response_from_llm(prompt: str) -> None:
    """Invoke the RAG chain (with streaming) and display the answer + sources."""
    try:
        chain = st.session_state.chain
        chat_history = trim_chat_history(
            st.session_state.chat_history,
            max_turns=st.session_state.get("max_history_turns", 20),
        )
        context_documents: List[Any] = []

        def answer_stream():
            for chunk in chain.stream(
                {"input": prompt, "chat_history": chat_history}
            ):
                if chunk.get("context"):
                    context_documents[:] = chunk["context"]
                if chunk.get("answer"):
                    yield chunk["answer"]

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                answer = st.write_stream(answer_stream())
            except Exception:
                # Fall back to a non-streaming invocation.
                result = chain.invoke({"input": prompt, "chat_history": chat_history})
                answer = result.get("answer", "")
                context_documents[:] = result.get("context", [])
                st.markdown(answer)

            if context_documents:
                with st.expander("**Source documents**"):
                    display_source_documents(context_documents)

        # Update the chat history kept in memory.
        st.session_state.chat_history = append_turn(
            chat_history,
            user_message=prompt,
            assistant_message=answer,
            max_turns=st.session_state.get("max_history_turns", 20),
        )

        st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        st.warning(f"{type(e).__name__}: {e}")
        logger.exception("Failed to get a response from the LLM")


# ---------------------------------------------------------------------------
# Chatbot main
# ---------------------------------------------------------------------------

def chatbot() -> None:
    """Render the full chatbot page (sidebar + chat)."""
    from ui.sidebar import sidebar_and_document_chooser, clear_chat_history

    init_session_state()
    sidebar_and_document_chooser()

    st.divider()
    col1, col2 = st.columns([7, 3])
    with col1:
        st.subheader("Chat with your data")
    with col2:
        st.button("Clear Chat History", on_click=clear_chat_history)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": dict_welcome_message[st.session_state.assistant_language],
            }
        ]
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        if "chain" not in st.session_state:
            st.info("Please create or load a vectorstore first.")
        elif not has_api_key():
            st.info(
                f"Please insert your {st.session_state.LLM_provider} API key to continue."
            )
        else:
            with st.spinner("Running..."):
                get_response_from_llm(prompt=prompt)