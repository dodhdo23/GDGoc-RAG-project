"""Query retriever built on embedder + FAISS vector DB."""

from __future__ import annotations

from src.embeddings import SentenceTransformerEmbedder
from src.vectordb import FaissVectorDB


class Retriever:
    """Retrieve top-k chunks for a query."""

    def __init__(self, embedder: SentenceTransformerEmbedder, vectordb: FaissVectorDB) -> None:
        self.embedder = embedder
        self.vectordb = vectordb

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Return ranked chunk text, scores, and metadata."""
        if not query.strip():
            raise ValueError("Query must not be empty.")

        query_embedding = self.embedder.embed_query(query)
        scores, indices = self.vectordb.search(query_embedding, top_k=top_k)

        results: list[dict] = []
        for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
            if idx < 0 or idx >= len(self.vectordb.chunks):
                continue
            chunk = self.vectordb.chunks[idx]
            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                }
            )
        return results
