"""Simple text chunking for retrieval."""

from __future__ import annotations

from src.utils import Chunk, Document


def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[tuple[str, int, int]]:
    """
    Split text into overlapping character-based chunks.

    Returns a list of tuples: (chunk_text, start_idx, end_idx).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    clean_text = text.strip()
    if not clean_text:
        return []

    chunks: list[tuple[str, int, int]] = []
    step = chunk_size - chunk_overlap
    start = 0
    n = len(clean_text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk_text = clean_text[start:end].strip()
        if chunk_text:
            chunks.append((chunk_text, start, end))
        if end == n:
            break
        start += step

    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Split documents into chunks while preserving source metadata."""
    output: list[Chunk] = []
    for doc_idx, doc in enumerate(documents):
        spans = split_text(
            text=doc.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for local_idx, (chunk_text, start, end) in enumerate(spans):
            metadata = dict(doc.metadata)
            metadata.update(
                {
                    "doc_id": doc_idx,
                    "chunk_id": len(output),
                    "chunk_in_doc": local_idx,
                    "char_start": start,
                    "char_end": end,
                }
            )
            output.append(Chunk(text=chunk_text, metadata=metadata))
    return output
