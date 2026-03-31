# RAG Reproduction: Baseline Retrieval (Stage 1)

## 오늘 구현 범위

- 문서 로딩 (`.txt`, `.pdf`)
- 텍스트 청킹
- 임베딩 생성 (sentence-transformers)
- FAISS 인덱싱/저장/로드
- 쿼리 기반 Top-k 검색 CLI

## 오늘 제외 범위

- CRAG / corrective retrieval
- reranker / hybrid retrieval
- 평가 지표(Recall@k, MRR, EM, F1 등)
- OpenAI API / 답변 생성 파이프라인

## 폴더 구조

```text
rag_reproduction/
├─ README.md
├─ requirements.txt
├─ data/
│  ├─ raw/
│  └─ processed/
├─ src/
│  ├─ loaders.py
│  ├─ chunking.py
│  ├─ embeddings.py
│  ├─ vectordb.py
│  ├─ retriever.py
│  ├─ pipeline.py
│  └─ utils.py
└─ scripts/
   ├─ build_index.py
   └─ search_demo.py
```

## 설치

```bash
cd rag_reproduction
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
```

## 데이터 준비

`data/raw/`에 `.txt` 또는 `.pdf` 파일을 넣습니다.

예시:

```text
data/raw/
├─ lecture1.txt
├─ notes.txt
└─ paper.pdf
```

## 인덱스 생성

```bash
python scripts/build_index.py --raw_dir data/raw --index_dir data/processed/index
```

옵션:

- `--chunk_size` (기본값: 500)
- `--chunk_overlap` (기본값: 50)
- `--model_name` (기본값: `sentence-transformers/all-MiniLM-L6-v2`)

## 검색 데모 실행

```bash
python scripts/search_demo.py --index_dir data/processed/index --top_k 5 --query "RAG의 핵심 개념은?"
```

또는 `--query` 없이 실행하면 입력 프롬프트에서 질의를 받을 수 있습니다.

## 인덱스 산출물

`data/processed/index/` 내부에 아래 파일이 저장됩니다.

- `index.faiss`: 벡터 인덱스
- `chunks.json`: 청크 텍스트 + 메타데이터

## 다음 단계 아이디어

다음 스테이지에서 CRAG-lite, 평가 파이프라인, 생성기(LLM) 연결을 순차적으로 확장할 수 있습니다.
