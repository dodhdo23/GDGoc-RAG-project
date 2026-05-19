# Self-RAG Reproduction

Self-RAG 논문(*Learning to Retrieve, Generate, and Critique through Self-Reflection*)의  
self-reflection 기반 inference 구조를 경량 재현하고, Baseline RAG와 성능을 비교하는 프로젝트.

> 전체 학습 구조(critic model 학습, reflection token 학습)는 구현하지 않는다.  
> LLM 프롬프트로 Relevant / Supported / NeedRetrieval을 판단하는 방식으로 대체한다.

---

## 원본 harness 코딩 대비 변경점

### 추가된 것

| 파일 | 내용 |
|---|---|
| `src/critique.py` | LLM 프롬프트 기반 답변 평가 (Relevant / Supported / NeedRetrieval) |
| `src/self_rag.py` | Self-RAG 파이프라인 — retrieve → generate → critique → 재검색 → 재생성 루프 |
| `scripts/load_squad.py` | SQuAD validation에서 N개 샘플 → `data/raw/`, `data/eval.json` 저장 |
| `scripts/compare_demo.py` | 단일 질문으로 두 시스템 결과 + 응답 시간 나란히 출력 |
| `main.py` | FastAPI 서버 — `/index`, `/query`, `/self-query`, `/status` |
| `CLAUDE.md` | 프로젝트 코딩 가이드라인 |

### 수정된 것

| 파일 | 변경 내용 |
|---|---|
| `src/generator.py` | `LLMAnswerGenerator` 추가 (OpenAI 기반, `BaselineAnswerGenerator`는 유지) |
| `src/pipeline.py` | `run_self_rag()` 추가 |
| `src/evaluation.py` | `answer_contains_gold`, `context_contains_gold` 추가, `compare_systems()` 계획서 표 항목에 맞게 재정의 |
| `scripts/evaluate_demo.py` | `--mode baseline/self_rag/both` 지원, 비교 표 출력 |
| `requirements.txt` | `openai`, `fastapi`, `uvicorn`, `datasets` 추가 |

### 삭제된 것

- `data/raw/test.txt` — 샘플 파일, SQuAD로 대체
- `data/processed/sample_eval.json` — 구버전 평가 포맷, `data/eval.json`으로 대체
- GitHub API 코드 전면 제거 (순수 RAG 연구 프로젝트)

---

## 프로젝트 구조

```
rag_reproduction/
├── CLAUDE.md
├── main.py                  FastAPI 서버
├── requirements.txt
├── data/
│   ├── raw/                 인덱싱할 문서 (load_squad.py가 여기에 저장)
│   ├── processed/
│   │   └── index/           FAISS index.faiss + chunks.json
│   └── eval.json            평가용 question/answer 쌍
├── src/
│   ├── utils.py             Document / Chunk 데이터클래스
│   ├── loaders.py           .txt / .pdf 로더
│   ├── chunking.py          슬라이딩 윈도우 청킹
│   ├── embeddings.py        SentenceTransformerEmbedder
│   ├── vectordb.py          FaissVectorDB
│   ├── retriever.py         Retriever
│   ├── generator.py         BaselineAnswerGenerator / LLMAnswerGenerator
│   ├── critique.py          AnswerCritique (Self-RAG 판단 모듈)
│   ├── self_rag.py          SelfRAG 파이프라인
│   ├── pipeline.py          run_baseline_rag / run_self_rag
│   └── evaluation.py        평가 지표 + compare_systems()
└── scripts/
    ├── load_squad.py        SQuAD 데이터 준비
    ├── build_index.py       FAISS 인덱스 빌드
    ├── search_demo.py       retrieval 단독 테스트
    ├── run_rag_demo.py      Baseline RAG 단일 실행
    ├── compare_demo.py      Baseline vs Self-RAG 단일 질문 비교
    └── evaluate_demo.py     배치 평가 + 비교 표 출력
```

---

## 설치

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`.env` 파일 생성:
```
OPENAI_API_KEY=sk-...
```

---

## 실행 순서

### 1단계 — 데이터 준비

```powershell
# 구현 테스트용 (50개)
python scripts\load_squad.py --n 50

# 최종 실험용 (200개)
python scripts\load_squad.py --n 200
```

### 2단계 — 인덱스 빌드

```powershell
python scripts\build_index.py
```

### 3단계 — 단일 질문 비교

```powershell
python scripts\compare_demo.py --query "Where is Stanford University located?"
```

### 4단계 — 배치 평가 및 비교 표 출력

```powershell
# Baseline만
python scripts\evaluate_demo.py --eval_file data/eval.json --mode baseline

# 두 시스템 비교
python scripts\evaluate_demo.py --eval_file data/eval.json --mode both
```

출력 예시:
```
=================================================================
항목                              Baseline RAG         Self-RAG
-----------------------------------------------------------------
평가 질문 수                              200              200
Top-k                                       3        3 → 부족 시 6
정답 포함률                            62.0%            71.0%
검색 성공률                            74.0%            82.0%
Hallucination 의심률                   18.0%            11.0%
평균 retrieval 횟수                     1.00회           1.34회
평균 재생성 횟수                         0.00회           0.34회
평균 응답 시간                           0.45초           2.31초
=================================================================
```

### FastAPI 서버

```powershell
uvicorn main:app --reload
```

| 엔드포인트 | 설명 |
|---|---|
| `POST /index` | data/raw/ 문서로 FAISS 인덱스 빌드 |
| `POST /query` | Baseline RAG 질문 |
| `POST /self-query` | Self-RAG 질문 |
| `GET /status` | 인덱스 상태 확인 |

---

## Self-RAG 구현 범위

```
Baseline RAG
  질문 → top-k 검색 → LLM 답변 생성 → 출력

Self-RAG (이 프로젝트)
  질문 → top-k 검색 → LLM 답변 생성
       → critique 판단 (Relevant / Supported / NeedRetrieval)
       → NeedRetrieval=True면 top-k×2로 재검색 → 재생성 (최대 2회)
       → 최종 답변 출력
```

논문 원본과의 차이:
- reflection token 학습 없음
- critic model 학습 없음
- LLM 프롬프트로 판단 로직 대체

---

## 평가 지표

| 지표 | 설명 |
|---|---|
| `answer_contains_gold` | 생성 답변에 정답이 포함되는지 |
| `context_contains_gold` | 검색된 문서에 정답이 포함되는지 (검색 성공률) |
| `grounded_token_recall` | 답변 토큰이 검색 문서에 근거하는지 (hallucination 프록시) |
| `token_f1` | 정답과의 토큰 단위 F1 |
| `retrieval_count` | 총 retrieval 횟수 |
| `regeneration_count` | 재생성 횟수 |

---

## 실험 결과 — Baseline RAG vs Self-RAG 케이스 비교

데이터셋: SQuAD validation (Super Bowl 50 관련 4개 문서, 30개 질문)  
모델: gpt-4o-mini / Embedder: all-MiniLM-L6-v2 / Top-k: 3

### Case 1 — 단일 문서 사실 질문

> **Query:** "Who headlined the Super Bowl 50 halftime show and who were the special guests?"  
> **Gold:** Coldplay, with Beyoncé and Bruno Mars

| 지표 | Baseline RAG | Self-RAG |
|---|---|---|
| token_f1 | 0.091 | **0.286** |
| grounded_token_recall | 1.000 | 1.000 |
| 응답 시간 | 3.78초 | 6.81초 |

**분석:** Baseline은 정답을 포함하지만 불필요한 문장(CBS 광고비, 팀 통계)을 함께 출력. Self-RAG는 critique 단계에서 관련 정보만 선별해 간결하게 답변.

---

### Case 2 — 다중 정보 통합 질문

> **Query:** "What was the final score and who was named MVP in Super Bowl 50?"  
> **Gold:** Denver Broncos won 24-10, Von Miller was named MVP

| 지표 | Baseline RAG | Self-RAG |
|---|---|---|
| token_f1 | 0.222 | **0.696** |
| grounded_token_recall | 1.000 | 0.786 |
| 응답 시간 | 3.67초 | 8.15초 |

**분석:** Baseline은 스코어 언급 없이 MVP 관련 context만 나열. Self-RAG는 서로 다른 chunk에서 스코어(squad_000)와 MVP(squad_002) 정보를 통합해 명확한 답변 생성. token_f1 기준 3배 이상 차이.

---

### Case 3 — 교차 문서 추론 질문

> **Query:** "How many times was Cam Newton sacked in Super Bowl 50 and what was his regular season record?"  
> **Gold:** Newton was sacked seven times, Panthers had a 15-1 regular season record

| 지표 | Baseline RAG | Self-RAG |
|---|---|---|
| token_f1 | 0.263 | **0.720** |
| grounded_token_recall | 1.000 | 0.929 |
| 응답 시간 | 4.04초 | 8.05초 |

**분석:** squad_001(정규시즌 기록)과 squad_002(새크 횟수) 두 문서를 동시에 참조해야 하는 질문. Baseline은 두 문서 내용을 그대로 나열(Broncos 12-4 기록 등 불필요한 정보 포함). Self-RAG는 필요한 정보만 추출해 "7번 새크, 15–1 기록"으로 정확히 답변.

---

### 종합 비교

| | Baseline RAG | Self-RAG |
|---|---|---|
| 평균 token_f1 | 0.192 | **0.567** |
| 평균 응답 시간 | ~3.8초 | ~7.7초 |
| 답변 스타일 | context 나열형 | 핵심 정보 선별형 |
| 다중 문서 통합 | 미흡 | 우수 |

- **Self-RAG token_f1이 약 3배 높음** → 답변 정확도 및 간결성 우위
- **grounded_token_recall은 Baseline이 더 높은 경우도 있음** → context를 통째로 복사했기 때문으로, 정보 선별 능력과는 무관
- **응답 시간은 Self-RAG가 약 2배 느림** → critique 루프(LLM 추가 호출) 비용
