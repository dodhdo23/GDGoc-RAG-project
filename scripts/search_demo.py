"""CLI demo script for top-k retrieval search."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import load_retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search top-k chunks from saved FAISS index.")
    parser.add_argument("--index_dir", type=str, default="data/processed/index", help="Path to saved index artifacts")
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model name",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--query", type=str, default=None, help="Search query (optional)")
    return parser.parse_args()


def preview(text: str, max_len: int = 180) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_len:
        return clean
    return f"{clean[:max_len]}..."


def main() -> None:
    args = parse_args()
    index_dir = PROJECT_ROOT / args.index_dir
    retriever = load_retriever(index_dir=index_dir, model_name=args.model_name)

    query = args.query.strip() if args.query else ""
    if not query:
        query = input("Enter your query: ").strip()
    if not query:
        raise ValueError("Query must not be empty.")

    results = retriever.retrieve(query=query, top_k=args.top_k)
    if not results:
        print("No results found.")
        return

    print(f"\nQuery: {query}")
    print(f"Top-{args.top_k} results\n")
    for item in results:
        meta = item["metadata"]
        source = meta.get("source", "unknown")
        print(f"[{item['rank']}] score={item['score']:.4f} source={source}")
        print(f"    {preview(item['text'])}")
        print("")


if __name__ == "__main__":
    main()
