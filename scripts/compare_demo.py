"""단일 질문으로 Baseline RAG vs Self-RAG 결과를 나란히 출력합니다."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.evaluation import answer_contains_gold, compare_systems, context_contains_gold
from src.pipeline import run_baseline_rag, run_self_rag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", type=str, default="data/processed/index")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_iter", type=int, default=2)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--gold_answer", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 .env에 없습니다.")

    index_dir = PROJECT_ROOT / args.index_dir
    query = args.query.strip() if args.query else input("질문: ").strip()
    if not query:
        raise ValueError("Query must not be empty.")

    print("\n[Baseline RAG 실행 중...]")
    t0 = time.time()
    baseline = run_baseline_rag(query=query, index_dir=index_dir, top_k=args.top_k)
    b_time = time.time() - t0

    print("[Self-RAG 실행 중...]")
    t0 = time.time()
    self_rag = run_self_rag(query=query, index_dir=index_dir, api_key=api_key, top_k=args.top_k, max_iter=args.max_iter)
    s_time = time.time() - t0

    print("\n" + "=" * 60)
    print(f"질문: {query}")
    print("=" * 60)
    print(f"\n[Baseline RAG]  ({b_time:.2f}초)")
    print(baseline["answer"])
    print(f"\n[Self-RAG]  ({s_time:.2f}초, retrieval {self_rag['retrieval_count']}회, 재생성 {self_rag['regeneration_count']}회)")
    print(self_rag["answer"])

    if args.gold_answer:
        result = compare_systems(baseline, self_rag, args.gold_answer, b_time, s_time)
        print("\n[평가 지표]")
        for key, val in result["baseline"]["metrics"].items():
            b_val = result["baseline"]["metrics"][key]
            s_val = result["self_rag"]["metrics"][key]
            print(f"  {key:<28} baseline={b_val:.3f}  self_rag={s_val:.3f}")


if __name__ == "__main__":
    main()
