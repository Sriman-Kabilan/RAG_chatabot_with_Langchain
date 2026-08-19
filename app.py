"""RAG chatbot - Streamlit entry point.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st

from ui.chat import chatbot

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Chat With Your Data")

st.title("🤖 RAG chatbot")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    chatbot()