"""FAISS-backed vector store for retrieval baseline."""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from src.utils import Chunk, chunk_from_dict, chunk_to_dict, load_json, save_json


class FaissVectorDB:
    """Minimal FAISS vector DB wrapper using inner-product similarity."""

    def __init__(self, index: faiss.Index | None = None, chunks: list[Chunk] | None = None) -> None:
        self.index = index
        self.chunks = chunks or []

    def build(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        """Build index from chunk embeddings and store associated chunks."""
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")
        if embeddings.shape[0] == 0:
            raise ValueError("Cannot build index with zero chunks")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        self.index = index
        self.chunks = chunks

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Search top-k nearest chunks for one query embedding."""
        if self.index is None:
            raise ValueError("Index not initialized. Build or load index first.")
        if query_embedding.ndim != 1:
            raise ValueError("query_embedding must be 1D")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        query = np.expand_dims(query_embedding.astype(np.float32), axis=0)
        scores, indices = self.index.search(query, top_k)
        return scores[0], indices[0]

    def save(self, index_dir: str | Path) -> None:
        """Save FAISS index and chunk metadata to local directory."""
        if self.index is None:
            raise ValueError("No index to save.")

        out_dir = Path(index_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out_dir / "index.faiss"))
        save_json([chunk_to_dict(c) for c in self.chunks], out_dir / "chunks.json")

    @classmethod
    def load(cls, index_dir: str | Path) -> "FaissVectorDB":
        """Load FAISS index and chunk metadata from local directory."""
        in_dir = Path(index_dir)
        index_path = in_dir / "index.faiss"
        chunks_path = in_dir / "chunks.json"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunk metadata not found: {chunks_path}")

        index = faiss.read_index(str(index_path))
        chunk_dicts = load_json(chunks_path)
        chunks = [chunk_from_dict(item) for item in chunk_dicts]
        return cls(index=index, chunks=chunks)
