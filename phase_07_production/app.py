"""
phase_07_production/app.py
=============================
The full AI Research Assistant, served as an HTTP API.

Run locally:  uvicorn phase_07_production.app:app --reload
Docs UI:      http://localhost:8000/docs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import shutil
import tempfile
from pathlib import Path as PathLib

from fastapi import FastAPI, UploadFile, File, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.document_loader import recursive_chunk, load_document, validate_file
from core.vector_store import get_vector_store, RetrievedChunk
from core.llm_client import LLMClient
from core.prompts import build_grounded_prompt
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from core.vector_store import warm_embedding_cache
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from core.session import check_upload_limit, record_upload
from core.session import (
    SESSION_COOKIE_NAME, new_session_id, check_rate_limit,
    record_query, clear_session_usage,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Warming embedding model cache at startup...")
    warm_embedding_cache()
    print("Warm-up complete — ready for requests.")
    yield

app = FastAPI(title="AI Research Assistant API", version="1.0", lifespan=lifespan)

# CORS — allows the Astro portfolio site (different origin) to call this API.
# Restrict allow_origins to your actual site domain before final deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://rambhupalpayyavula.com",
        "http://localhost:4321",  # Astro's local dev server, for testing the site against this API locally
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = LLMClient()


def get_or_create_session(request: Request, response: Response) -> str:
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_id:
        session_id = new_session_id()
        response.set_cookie(SESSION_COOKIE_NAME, session_id, httponly=True, samesite="lax", max_age=86400)
    return session_id


# ── Request/response models ──────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


class UploadResponse(BaseModel):
    filename: str
    chunks_ingested: int
    status: str


# ── Endpoints ─────────────────────────────────────────────────────────────
@app.get("/")
def serve_ui():
    static_dir = Path(__file__).resolve().parent / "static"
    return FileResponse(static_dir / "index.html")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload", response_model=UploadResponse)
@app.post("/upload", response_model=UploadResponse)
async def upload_document(request: Request, response: Response, file: UploadFile = File(...)):
    session_id = get_or_create_session(request, response)

    with tempfile.NamedTemporaryFile(delete=False, suffix=PathLib(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = PathLib(tmp.name)

    try:
        size_mb = tmp_path.stat().st_size / (1024 * 1024)
        if size_mb > 20:
            raise HTTPException(413, f"File too large: {size_mb:.1f}MB (max 20MB per file)")

        # NEW — cumulative session check
        allowed, reason = check_upload_limit(session_id, tmp_path.stat().st_size)
        if not allowed:
            raise HTTPException(413, reason)

        if PathLib(file.filename).suffix.lower() not in {".pdf", ".docx", ".txt", ".md"}:
            raise HTTPException(400, f"Unsupported file type: {PathLib(file.filename).suffix}")

        real_path = tmp_path.with_suffix(PathLib(file.filename).suffix)
        tmp_path.rename(real_path)

        text = load_document(real_path)
        if not text.strip():
            raise HTTPException(422, "No extractable text found in this file")

        chunks = recursive_chunk(text, source=file.filename)
        store = get_vector_store(session_id)
        store.upsert(
            ids=[f"{session_id}_{file.filename}_{c.chunk_index}" for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "chunk_index": c.chunk_index} for c in chunks],
        )

        record_upload(session_id, real_path.stat().st_size)  # NEW

        return UploadResponse(filename=file.filename, chunks_ingested=len(chunks), status="ok")

    finally:
        real_path.unlink(missing_ok=True) if 'real_path' in dir() else tmp_path.unlink(missing_ok=True)

@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest, request: Request, response: Response):
    session_id = get_or_create_session(request, response)

    allowed, reason = check_rate_limit(session_id)
    if not allowed:
        raise HTTPException(429, reason)

    store = get_vector_store(session_id)
    retrieved: list[RetrievedChunk] = store.query(payload.question, top_k=5)

    if not retrieved:
        return AskResponse(answer="No documents have been uploaded yet, or nothing relevant was found.", sources=[])

    system_prompt = build_grounded_prompt(retrieved)
    answer = llm.simple(system=system_prompt, user_message=payload.question, temperature=0.0)
    record_query(session_id)

    return AskResponse(
        answer=answer,
        sources=[f"{c.source} (relevance: {c.relevance_score:.2f})" for c in retrieved],
    )

@app.post("/clear")
def clear_session(request: Request, response: Response):
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_id:
        return {"status": "no active session"}

    store = get_vector_store(session_id)
    store.delete_all()
    clear_session_usage(session_id)
    response.delete_cookie(SESSION_COOKIE_NAME)
    return {"status": "session cleared"}
