"""CLI script to build retrieval index from raw documents."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import build_and_save_index, create_chunks, load_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index from raw documents.")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Directory containing .txt/.pdf files")
    parser.add_argument("--index_dir", type=str, default="data/processed/index", help="Output directory for index")
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model name",
    )
    parser.add_argument("--chunk_size", type=int, default=500, help="Character chunk size")
    parser.add_argument("--chunk_overlap", type=int, default=50, help="Character chunk overlap")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = PROJECT_ROOT / args.raw_dir
    index_dir = PROJECT_ROOT / args.index_dir

    documents = load_documents(raw_dir)
    if not documents:
        raise RuntimeError(f"No documents found in: {raw_dir}")

    chunks = create_chunks(
        documents=documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    if not chunks:
        raise RuntimeError("No chunks generated. Check document contents and chunk settings.")

    build_and_save_index(
        chunks=chunks,
        index_dir=index_dir,
        model_name=args.model_name,
    )

    print("[DONE] Index build completed.")
    print(f"  - Documents: {len(documents)}")
    print(f"  - Chunks: {len(chunks)}")
    print(f"  - Index path: {index_dir}")


if __name__ == "__main__":
    main()
