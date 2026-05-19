# CLAUDE.md — Self-RAG Reproduction Project

## 프로젝트 목적

Self-RAG 논문(Learning to Retrieve, Generate, and Critique through Self-Reflection)의
inference 구조를 경량 재현하고, Baseline RAG와 성능을 비교한다.

학습 기반 critic model은 구현하지 않는다.
LLM 프롬프트로 Relevant / Supported / NeedRetrieval을 판단하는 방식으로 대체한다.

---

## 기술 스택

| 역할 | 도구 |
|---|---|
| 임베딩 | sentence-transformers/all-MiniLM-L6-v2 |
| 벡터 DB | FAISS (IndexFlatIP) |
| LLM | OpenAI gpt-4o-mini |
| 데이터셋 | SQuAD validation (100개) |
| 서버 | FastAPI |

---

## 파일 구조

```
src/
  utils.py       Document / Chunk 데이터클래스, JSON 유틸
  loaders.py     .txt / .pdf 로더 (metadata key: "source")
  chunking.py    슬라이딩 윈도우 청킹
  embeddings.py  SentenceTransformerEmbedder
  vectordb.py    FaissVectorDB (Windows 경로 처리 포함)
  retriever.py   Retriever.retrieve(query, top_k) → list[dict]
  generator.py   BaselineAnswerGenerator (extractive, API 불필요)
                 LLMAnswerGenerator (OpenAI 기반)
  critique.py    AnswerCritique (LLM 프롬프트로 3가지 판단)
  self_rag.py    SelfRAG.run() — retrieve→generate→critique 루프
  pipeline.py    run_baseline_rag(), run_self_rag() 고수준 함수
  evaluation.py  token_f1, exact_match, compare_systems()

scripts/
  load_squad.py     SQuAD 다운로드 → data/raw/, data/eval.json
  build_index.py    data/raw/ → FAISS 인덱스 → data/processed/index/
  run_rag_demo.py   Baseline RAG CLI
  compare_demo.py   Baseline RAG vs Self-RAG 비교 CLI
  evaluate_demo.py  eval.json 기반 지표 평가

main.py   FastAPI 서버 (/index, /query, /self-query, /status)
```

---

## 데이터 디렉토리 규칙

```
data/
  raw/           인덱싱할 .txt/.pdf 문서 (load_squad.py가 여기에 저장)
  processed/
    index/       FAISS index.faiss + chunks.json
  eval.json      평가용 question/answer 쌍
```

---

## 실행 순서

```bash
# 1. 데이터 준비
python scripts/load_squad.py

# 2. 인덱스 빌드
python scripts/build_index.py

# 3. 비교 실행
python scripts/compare_demo.py --query "Where is Stanford located?"

# 4. 평가
python scripts/evaluate_demo.py --eval_file data/eval.json
```

---

## 코딩 규칙

### 1. 수정 전 멈추고 생각하기

- 가정은 명시적으로 말한다. 불확실하면 먼저 질문한다.
- 여러 해석이 가능하면 조용히 선택하지 말고 제시한다.
- 더 단순한 방법이 있으면 말한다.

### 2. 최소한으로 짜기

- 요청된 것만 구현한다. 추측성 기능 없음.
- 단일 사용 코드에 추상화 계층 추가 금지.
- 50줄로 가능한 걸 200줄로 쓰면 다시 짠다.

### 3. 건드리지 말아야 할 것

`src/utils.py`, `src/chunking.py`, `src/loaders.py`,
`src/embeddings.py`, `src/vectordb.py`, `src/retriever.py`
→ 잘 동작하고 있다. 수정 이유가 없으면 손대지 않는다.

기존 스크립트(`build_index.py`, `search_demo.py`, `run_rag_demo.py`, `evaluate_demo.py`)
→ 인터페이스 유지. 내부 수정 필요하면 명시적으로 말한다.

### 4. 검증 가능한 목표로 변환

```
작업 → 검증 방법
인덱스 빌드 → build_index.py 실행 후 data/processed/index/ 생성 확인
Baseline RAG → run_rag_demo.py로 답변 출력 확인
Self-RAG → compare_demo.py로 두 시스템 결과 비교 확인
평가 → evaluate_demo.py 실행 후 token_f1 수치 확인
```

---

## 주의사항

- OPENAI_API_KEY는 `.env`에 저장. 코드에 하드코딩 금지.
- GitHub API 없음. 이 프로젝트는 순수 RAG 연구용.
- Windows 환경에서 비ASCII 경로 문제 있음 → `vectordb.py`의 `_faiss_safe_path` 참고.
