"""Embedding wrapper based on sentence-transformers."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder:
    """Small wrapper for sentence-transformers embedding model."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load the embedding model. On the first run, "
                "sentence-transformers may need internet access to download the model, "
                "or you can pass a local model path via --model_name. "
                f"Requested model: {model_name}"
            ) from exc

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a list of texts and return float32 numpy array."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        vectors = self.model.encode(
            list(texts),
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        vectors = self.embed_texts([query])
        if vectors.size == 0:
            raise ValueError("Failed to embed query.")
        return vectors[0]
