"""Baseline RAG / Self-RAG 배치 평가 및 비교 표 출력.

사용 예:
  python scripts/evaluate_demo.py --eval_file data/eval.json --mode baseline
  python scripts/evaluate_demo.py --eval_file data/eval.json --mode both
  python scripts/evaluate_demo.py --eval_file data/eval.json --mode both --groundedness
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from statistics import mean
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.evaluation import (
    answer_contains_gold,
    context_contains_gold,
    groundedness_score,
    grounded_token_recall,
    load_eval_data,
)
from src.pipeline import run_baseline_rag, run_self_rag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str, required=True)
    parser.add_argument("--index_dir", type=str, default="data/processed/index")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_iter", type=int, default=2)
    parser.add_argument("--mode", choices=["baseline", "self_rag", "both"], default="baseline")
    parser.add_argument("--groundedness", action="store_true", help="LLM 기반 groundedness 평가 (API 호출 발생)")
    return parser.parse_args()


def run_baseline(examples: list[dict], index_dir: Path, top_k: int) -> tuple[list[dict], float]:
    results, total_time = [], 0.0
    for ex in examples:
        t0 = time.time()
        r = run_baseline_rag(query=ex["question"], index_dir=index_dir, top_k=top_k)
        total_time += time.time() - t0
        results.append({"result": r, "gold": ex["answer"]})
    return results, total_time / len(examples)


def run_self(examples: list[dict], index_dir: Path, top_k: int, max_iter: int, api_key: str) -> tuple[list[dict], float]:
    results, total_time = [], 0.0
    for ex in examples:
        t0 = time.time()
        r = run_self_rag(query=ex["question"], index_dir=index_dir, api_key=api_key, top_k=top_k, max_iter=max_iter)
        total_time += time.time() - t0
        results.append({"result": r, "gold": ex["answer"]})
    return results, total_time / len(examples)


def compute_metrics(results: list[dict], key_chunks: str, api_key: str | None, use_groundedness: bool) -> dict:
    acg, ccg, ret_count, regen_count = [], [], [], []
    grounded, partially, ungrounded = 0, 0, 0

    for item in results:
        r, gold = item["result"], item["gold"]
        chunks = r.get(key_chunks, [])
        acg.append(answer_contains_gold(r["answer"], gold))
        ccg.append(context_contains_gold(chunks, gold))
        ret_count.append(float(r.get("retrieval_count", 1)))
        regen_count.append(float(r.get("regeneration_count", 0)))

        if use_groundedness and api_key:
            label = groundedness_score(r["query"], r["answer"], chunks, api_key)
            if label == "GROUNDED":
                grounded += 1
            elif label == "PARTIALLY_GROUNDED":
                partially += 1
            else:
                ungrounded += 1

    n = len(results)
    metrics: dict = {
        "retrieval_hit_rate": mean(ccg),
        "answer_contains_match": mean(acg),
        "avg_retrieval_count": mean(ret_count),
        "avg_regeneration_count": mean(regen_count),
        "regeneration_rate": mean(1.0 if v > 0 else 0.0 for v in regen_count),
    }

    if use_groundedness and api_key:
        metrics["grounded_rate"] = (grounded + partially) / n
        metrics["hallucination_rate"] = ungrounded / n
    else:
        # fallback: lexical proxy
        gtr = [grounded_token_recall(item["result"]["answer"], item["result"].get(key_chunks, [])) for item in results]
        metrics["grounded_rate"] = mean(1.0 if v >= 0.3 else 0.0 for v in gtr)
        metrics["hallucination_rate"] = mean(1.0 if v < 0.3 else 0.0 for v in gtr)

    return metrics


def print_table(n: int, top_k: int, b: dict | None, b_time: float, s: dict | None, s_time: float, use_groundedness: bool) -> None:
    groundedness_note = "" if use_groundedness else " (lexical proxy)"

    def pct(v: float | None) -> str:
        return f"{v*100:.1f}%" if v is not None else "  -  "

    def cnt(v: float | None) -> str:
        return f"{v:.2f}회" if v is not None else "  -  "

    def sec(v: float) -> str:
        return f"{v:.2f}초" if v > 0 else "  -  "

    W = 65
    print("\n" + "=" * W)
    print(f"{'평가 지표':<30} {'Baseline RAG':>15} {'Self-RAG':>15}")
    print("-" * W)
    print(f"{'평가 질문 수':<30} {n:>15} {n:>15}")
    print(f"{'Top-k':<30} {str(top_k):>15} {'3 → 부족 시 6':>15}")
    print("-" * W)
    print(f"{'Retrieval Hit Rate':<30} {pct(b['retrieval_hit_rate'] if b else None):>15} {pct(s['retrieval_hit_rate'] if s else None):>15}")
    print(f"{'Answer Contains Match':<30} {pct(b['answer_contains_match'] if b else None):>15} {pct(s['answer_contains_match'] if s else None):>15}")
    print(f"{'Grounded Answer 비율{}'.format(groundedness_note):<30} {pct(b['grounded_rate'] if b else None):>15} {pct(s['grounded_rate'] if s else None):>15}")
    print(f"{'Hallucination Rate{}'.format(groundedness_note):<30} {pct(b['hallucination_rate'] if b else None):>15} {pct(s['hallucination_rate'] if s else None):>15}")
    print("-" * W)
    print(f"{'평균 Retrieval 횟수':<30} {cnt(b['avg_retrieval_count'] if b else None):>15} {cnt(s['avg_retrieval_count'] if s else None):>15}")
    print(f"{'Regeneration 발생률':<30} {pct(b['regeneration_rate'] if b else None):>15} {pct(s['regeneration_rate'] if s else None):>15}")
    print(f"{'평균 응답 시간':<30} {sec(b_time):>15} {sec(s_time):>15}")
    print("=" * W)
    if not use_groundedness:
        print("  * Grounded/Hallucination은 lexical proxy 기준 (--groundedness 플래그로 LLM 판단 가능)")


def main() -> None:
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY")
    index_dir = PROJECT_ROOT / args.index_dir
    examples = load_eval_data(str(PROJECT_ROOT / args.eval_file))
    n = len(examples)

    if args.groundedness and not api_key:
        raise RuntimeError("--groundedness 사용 시 OPENAI_API_KEY가 필요합니다.")
    if args.mode in ("self_rag", "both") and not api_key:
        raise RuntimeError("Self-RAG 실행 시 OPENAI_API_KEY가 필요합니다.")

    b_metrics, b_time = None, 0.0
    s_metrics, s_time = None, 0.0

    if args.mode in ("baseline", "both"):
        print(f"[Baseline RAG] {n}개 평가 중...")
        b_results, b_time = run_baseline(examples, index_dir, args.top_k)
        b_metrics = compute_metrics(b_results, "retrieved_chunks", api_key, args.groundedness)

    if args.mode in ("self_rag", "both"):
        print(f"[Self-RAG] {n}개 평가 중...")
        s_results, s_time = run_self(examples, index_dir, args.top_k, args.max_iter, api_key)
        s_metrics = compute_metrics(s_results, "chunks", api_key, args.groundedness)

    print_table(n, args.top_k, b_metrics, b_time, s_metrics, s_time, args.groundedness)


if __name__ == "__main__":
    main()
