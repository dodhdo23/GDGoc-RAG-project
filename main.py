import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

INDEX_DIR = Path("data/processed/index")

app = FastAPI(title="Self-RAG Reproduction API")


# ── Request models ────────────────────────────────────────────────────────────

class IndexRequest(BaseModel):
    data_dir: str = "data"
    chunk_size: int = 500
    chunk_overlap: int = 50


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


class SelfQueryRequest(BaseModel):
    query: str
    top_k: int = 3
    max_iter: int = 2


# ── Exception handler ─────────────────────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=400, content={"detail": "Invalid request body."})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_index():
    if not (INDEX_DIR / "index.faiss").exists():
        raise HTTPException(status_code=404, detail="인덱스가 없습니다. POST /index를 먼저 실행하세요.")


def _require_api_key():
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY가 설정되어 있지 않습니다.")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/index")
def build_index(body: IndexRequest) -> Dict[str, Any]:
    """data/ 폴더 문서로 FAISS 인덱스를 빌드합니다."""
    from src.pipeline import build_and_save_index, create_chunks, load_documents

    try:
        documents = load_documents(Path(body.data_dir))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not documents:
        raise HTTPException(status_code=400, detail=f"'{body.data_dir}'에 .txt/.pdf 파일이 없습니다.")

    chunks = create_chunks(documents, chunk_size=body.chunk_size, chunk_overlap=body.chunk_overlap)
    build_and_save_index(chunks, index_dir=INDEX_DIR)

    return {"status": "ok", "documents": len(documents), "chunks": len(chunks)}


@app.post("/query")
def baseline_query(body: QueryRequest) -> Dict[str, Any]:
    """Baseline RAG: 검색 후 답변 생성."""
    _require_index()
    from src.pipeline import run_baseline_rag

    try:
        result = run_baseline_rag(query=body.query, index_dir=INDEX_DIR, top_k=body.top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "query": result["query"],
        "answer": result["answer"],
        "chunks": [
            {"rank": c["rank"], "score": round(c["score"], 4), "text": c["text"]}
            for c in result["retrieved_chunks"]
        ],
    }


@app.post("/self-query")
def self_rag_query(body: SelfQueryRequest) -> Dict[str, Any]:
    """Self-RAG: retrieve → generate → critique → (re-retrieve → regenerate) 반복."""
    _require_index()
    _require_api_key()
    from src.pipeline import run_self_rag

    try:
        result = run_self_rag(
            query=body.query,
            index_dir=INDEX_DIR,
            api_key=OPENAI_API_KEY,
            top_k=body.top_k,
            max_iter=body.max_iter,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "query": result["query"],
        "answer": result["answer"],
        "retrieval_count": result["retrieval_count"],
        "regeneration_count": result["regeneration_count"],
        "chunks": [
            {"rank": c["rank"], "score": round(c["score"], 4), "text": c["text"]}
            for c in result["chunks"]
        ],
    }


@app.get("/status")
def status() -> Dict[str, Any]:
    """인덱스 상태 확인."""
    from src.utils import load_json

    index_ready = (INDEX_DIR / "index.faiss").exists()
    chunk_count = 0
    if (INDEX_DIR / "chunks.json").exists():
        try:
            chunk_count = len(load_json(INDEX_DIR / "chunks.json"))
        except Exception:
            pass

    return {"index_ready": index_ready, "chunk_count": chunk_count}
