"""Entrypoint de la API FastAPI."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse

from app.config import get_settings
from app.middleware.logging import RequestContextMiddleware, configure_logging
from app.routers import agents, chat, embeddings, health, rag
from app.routers import models as models_router
from app.services.llm_client import LLMClient

STATIC_DIR = Path(__file__).parent / "static"

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.api_log_level)

    client = LLMClient(
        base_url=settings.litellm_base_url,
        master_key=settings.litellm_master_key,
        timeout=settings.litellm_request_timeout,
    )
    app.state.llm = client
    app.state.rag = None
    app.state.watcher = None

    # Inicializar RAG (Qdrant + indexer + retriever). Si Qdrant no está, app sigue vivo
    # pero los endpoints /v1/rag/* devolverán 503.
    try:
        from app.rag.embedder import Embedder
        from app.rag.indexer import Indexer
        from app.rag.retriever import Retriever
        from app.rag.store import QdrantStore

        store = QdrantStore(
            url=settings.qdrant_host,
            collection=settings.rag_collection,
            embed_dim=settings.rag_embed_dim,
        )
        await store.ensure_collection()
        embedder = Embedder(client=client, model=settings.default_embed_model)
        indexer = Indexer(
            store=store,
            embedder=embedder,
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
        )
        retriever = Retriever(store=store, embedder=embedder)
        app.state.rag = {"store": store, "indexer": indexer, "retriever": retriever}
        log.info("rag.ready", extra={"qdrant": settings.qdrant_host, "collection": settings.rag_collection})

        # Watcher opcional
        if settings.rag_watcher_enabled:
            from app.rag.watcher import DocsWatcher
            w = DocsWatcher(
                docs_dir=Path(settings.rag_docs_dir),
                indexer=indexer,
            )
            await w.start()
            app.state.watcher = w
            log.info("rag.watcher_started", extra={"dir": settings.rag_docs_dir})
    except Exception as e:
        log.warning("rag.init_failed", extra={"err": str(e)})

    log.info("api.startup", extra={"litellm_base_url": settings.litellm_base_url})
    try:
        yield
    finally:
        if app.state.watcher:
            await app.state.watcher.stop()
        if app.state.rag and app.state.rag.get("store") is not None:
            await app.state.rag["store"].aclose()
        await client.aclose()
        log.info("api.shutdown")


app = FastAPI(
    title="llm_ops API",
    description="Wrapper OpenAI-compatible delante de LiteLLM (Ollama/OpenAI/Anthropic), con RAG sobre Qdrant.",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(RequestContextMiddleware)


def verify_api_key(request: Request) -> None:
    settings = get_settings()
    if not settings.api_require_auth:
        return
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
        )
    token = auth.split(" ", 1)[1].strip()
    if token not in settings.api_keys_list:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid api key",
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail if isinstance(exc.detail, str) else str(exc.detail),
                "type": "invalid_request_error" if exc.status_code < 500 else "api_error",
                "code": exc.status_code,
            }
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log.exception("api.unhandled_error", extra={"path": request.url.path})
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "internal server error",
                "type": "api_error",
                "code": 500,
            }
        },
    )


# ---- UI estática (mismo origen que la API: sin CORS) ----

@app.get("/", include_in_schema=False)
async def index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(index_path, media_type="text/html")


# ---- Routers ----

app.include_router(health.router)

app.include_router(chat.router, prefix="/v1", dependencies=[Depends(verify_api_key)])
app.include_router(embeddings.router, prefix="/v1", dependencies=[Depends(verify_api_key)])
app.include_router(models_router.router, prefix="/v1", dependencies=[Depends(verify_api_key)])
app.include_router(rag.router, prefix="/v1", dependencies=[Depends(verify_api_key)])
app.include_router(agents.router, prefix="/v1", dependencies=[Depends(verify_api_key)])
