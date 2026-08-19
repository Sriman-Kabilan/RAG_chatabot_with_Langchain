"""Document loading, temporary file handling, and text splitting utilities."""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import List, Sequence

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    CSVLoader,
    DirectoryLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, TMP_DIR

logger = logging.getLogger(__name__)


def ensure_temp_dir(temp_dir: Path = TMP_DIR) -> None:
    """Create the temporary upload directory if it does not exist."""
    temp_dir.mkdir(parents=True, exist_ok=True)


def delete_temp_files(temp_dir: Path = TMP_DIR) -> None:
    """Delete all files inside the temporary upload directory."""
    if not temp_dir.exists():
        return
    for file_path in glob.glob(temp_dir.as_posix() + "/*"):
        try:
            os.remove(file_path)
            logger.debug("Deleted temporary file: %s", file_path)
        except OSError as exc:  # pragma: no cover - defensive
            logger.warning("Could not delete %s: %s", file_path, exc)


def save_uploaded_files(uploaded_files: Sequence, temp_dir: Path = TMP_DIR) -> None:
    """Write uploaded Streamlit files into the temporary directory."""
    ensure_temp_dir(temp_dir)
    for uploaded_file in uploaded_files:
        target = temp_dir / uploaded_file.name
        target.write_bytes(uploaded_file.getvalue())
        logger.info("Saved uploaded file: %s", uploaded_file.name)


def langchain_document_loader(temp_dir: Path = TMP_DIR) -> List[Document]:
    """Load PDF, TXT, CSV and DOCX documents from ``temp_dir``.

    Returns:
        A list of LangChain ``Document`` objects.
    """
    documents: List[Document] = []

    txt_loader = DirectoryLoader(
        temp_dir.as_posix(), glob="**/*.txt", loader_cls=TextLoader, show_progress=True
    )
    documents.extend(txt_loader.load())

    pdf_loader = DirectoryLoader(
        temp_dir.as_posix(), glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
    )
    documents.extend(pdf_loader.load())

    csv_loader = DirectoryLoader(
        temp_dir.as_posix(),
        glob="**/*.csv",
        loader_cls=CSVLoader,
        show_progress=True,
        loader_kwargs={"encoding": "utf8"},
    )
    documents.extend(csv_loader.load())

    doc_loader = DirectoryLoader(
        temp_dir.as_posix(),
        glob="**/*.docx",
        loader_cls=Docx2txtLoader,
        show_progress=True,
    )
    documents.extend(doc_loader.load())

    logger.info("Loaded %d documents from %s", len(documents), temp_dir)
    return documents


def split_documents_to_chunks(
    documents: Sequence[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """Split documents into overlapping chunks.

    Uses ``RecursiveCharacterTextSplitter`` which keeps paragraphs, sentences,
    and words together as much as possible.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info("Split %d documents into %d chunks", len(documents), len(chunks))
    return chunks