"""High-level retrieval pipeline helpers."""

from __future__ import annotations

from pathlib import Path

from src.chunking import chunk_documents
from src.embeddings import SentenceTransformerEmbedder
from src.loaders import load_documents_from_folder
from src.retriever import Retriever
from src.utils import Chunk, Document
from src.vectordb import FaissVectorDB


def load_documents(raw_data_dir: str | Path) -> list[Document]:
    """Load source documents from raw data directory."""
    return load_documents_from_folder(raw_data_dir)


def create_chunks(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Chunk loaded documents."""
    return chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def build_and_save_index(
    chunks: list[Chunk],
    index_dir: str | Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> None:
    """Embed chunks, build FAISS index, and save artifacts."""
    if not chunks:
        raise ValueError("No chunks provided for indexing.")

    embedder = SentenceTransformerEmbedder(model_name=model_name)
    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.embed_texts(texts)

    vectordb = FaissVectorDB()
    vectordb.build(embeddings=embeddings, chunks=chunks)
    vectordb.save(index_dir=index_dir)


def load_retriever(
    index_dir: str | Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Retriever:
    """Load retriever from saved index artifacts."""
    embedder = SentenceTransformerEmbedder(model_name=model_name)
    vectordb = FaissVectorDB.load(index_dir=index_dir)
    return Retriever(embedder=embedder, vectordb=vectordb)


def run_retrieval(
    query: str,
    index_dir: str | Path,
    top_k: int = 5,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> list[dict]:
    """Convenience function for one-shot retrieval."""
    retriever = load_retriever(index_dir=index_dir, model_name=model_name)
    return retriever.retrieve(query=query, top_k=top_k)
