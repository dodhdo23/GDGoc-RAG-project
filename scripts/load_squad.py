"""SQuAD validation set에서 일부를 가져와 저장합니다.

출력:
  data/raw/squad_*.txt  → 인덱싱용 문서 (build_index.py에서 사용)
  data/eval.json        → 평가용 question/answer 쌍 (evaluate_demo.py에서 사용)

권장 사용:
  처음 구현 테스트: python scripts/load_squad.py --n 50
  최종 실험:        python scripts/load_squad.py --n 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="가져올 QA 예제 수 (기본값: 200)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import load_dataset

    print(f"SQuAD validation에서 {args.n}개 로드 중...")
    ds = load_dataset("squad", split=f"validation[:{args.n}]")

    raw_dir = PROJECT_ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # context → .txt 파일 (중복 제거, context 하나 = 파일 하나)
    seen: set[str] = set()
    doc_count = 0
    for item in ds:
        context = item["context"].strip()
        if context in seen:
            continue
        seen.add(context)
        (raw_dir / f"squad_{doc_count:03d}.txt").write_text(context, encoding="utf-8")
        doc_count += 1

    # question + answer → eval.json
    eval_data = [
        {
            "question": item["question"],
            "answer": item["answers"]["text"][0],
        }
        for item in ds
    ]
    eval_path = PROJECT_ROOT / "data" / "eval.json"
    eval_path.write_text(json.dumps(eval_data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"문서 {doc_count}개 → data/raw/squad_*.txt")
    print(f"평가 데이터 {len(eval_data)}개 → data/eval.json")
    print("\n다음 단계:")
    print("  python scripts/build_index.py")
    print("  python scripts/evaluate_demo.py --eval_file data/eval.json --mode both")


if __name__ == "__main__":
    main()
