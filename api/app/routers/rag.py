"""Endpoints RAG: documents CRUD + query."""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.models import (
    RagDocument,
    RagDocumentsResponse,
    RagHit,
    RagIndexResponse,
    RagQueryRequest,
    RagQueryResponse,
)
from app.rag.parsers import SUPPORTED_EXTENSIONS, UnsupportedFormatError

log = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["rag"])


def _ensure_rag(request: Request):
    """Devuelve (indexer, retriever, store) o lanza 503 si RAG no está listo."""
    rag = getattr(request.app.state, "rag", None)
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG no inicializado (Qdrant no alcanzable?)")
    return rag


@router.get("/documents", response_model=RagDocumentsResponse)
async def list_documents(request: Request):
    rag = _ensure_rag(request)
    docs = await rag["store"].list_documents()
    return RagDocumentsResponse(
        data=[RagDocument(**d) for d in docs]
    )


@router.post("/documents", response_model=RagIndexResponse)
async def upload_document(request: Request, file: UploadFile = File(...)):
    rag = _ensure_rag(request)
    if not file.filename:
        raise HTTPException(status_code=400, detail="filename requerido")
    from pathlib import Path
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"extensión {ext!r} no soportada (soportadas: {sorted(SUPPORTED_EXTENSIONS)})",
        )
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="archivo vacío")
    try:
        result = await rag["indexer"].index_bytes(data, source=f"upload://{file.filename}")
    except UnsupportedFormatError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return RagIndexResponse(**result.__dict__)


@router.delete("/documents/{doc_id}")
async def delete_document(request: Request, doc_id: str):
    rag = _ensure_rag(request)
    n = await rag["store"].delete_doc(doc_id)
    if n == 0:
        raise HTTPException(status_code=404, detail="doc_id no encontrado")
    return {"deleted_chunks": n, "doc_id": doc_id}


@router.post("/query", response_model=RagQueryResponse)
async def query(request: Request, body: RagQueryRequest):
    rag = _ensure_rag(request)
    res = await rag["retriever"].query(body.query, top_k=body.top_k)
    return RagQueryResponse(
        query=res.query,
        hits=[RagHit(**h.__dict__) for h in res.hits],
        system_message=res.as_system_message(),
    )


@router.post("/reindex")
async def reindex_all(request: Request):
    """Reindexa todos los archivos de RAG_DOCS_DIR."""
    rag = _ensure_rag(request)
    from pathlib import Path
    from app.config import get_settings
    s = get_settings()
    docs_dir = Path(s.rag_docs_dir)
    if not docs_dir.exists():
        raise HTTPException(status_code=400, detail=f"docs_dir no existe: {docs_dir}")
    indexed = 0
    failed: list[dict] = []
    for path in docs_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            await rag["indexer"].index_path(path)
            indexed += 1
        except Exception as e:
            failed.append({"path": str(path), "error": str(e)})
    return {"indexed": indexed, "failed": failed}
