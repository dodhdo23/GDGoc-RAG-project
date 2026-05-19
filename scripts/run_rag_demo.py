"""CLI demo script for end-to-end baseline RAG."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import run_baseline_rag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an end-to-end baseline RAG demo.")
    parser.add_argument("--index_dir", type=str, default="data/processed/index", help="Path to saved index artifacts")
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model name",
    )
    parser.add_argument("--top_k", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument("--max_sentences", type=int, default=3, help="Maximum sentences in generated answer")
    parser.add_argument("--query", type=str, default=None, help="Question for the RAG pipeline")
    return parser.parse_args()


def preview(text: str, max_len: int = 180) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_len:
        return clean
    return f"{clean[:max_len]}..."


def main() -> None:
    args = parse_args()
    index_dir = PROJECT_ROOT / args.index_dir

    query = args.query.strip() if args.query else ""
    if not query:
        query = input("Enter your question: ").strip()
    if not query:
        raise ValueError("Query must not be empty.")

    result = run_baseline_rag(
        query=query,
        index_dir=index_dir,
        top_k=args.top_k,
        model_name=args.model_name,
        max_sentences=args.max_sentences,
    )

    print(f"\nQuestion: {result['query']}")
    print("\nGenerated answer:")
    print(result["answer"])
    print("\nRetrieved evidence:\n")

    for item in result["retrieved_chunks"]:
        source = item["metadata"].get("source", "unknown")
        print(f"[{item['rank']}] score={item['score']:.4f} source={source}")
        print(f"    {preview(item['text'])}")
        print("")


if __name__ == "__main__":
    main()
