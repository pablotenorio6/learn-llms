"""Entrypoint de la API FastAPI."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse

from app.config import get_settings
from app.middleware.logging import RequestContextMiddleware, configure_logging
from app.observability import init_tracer, shutdown_tracer
from app.routers import agents, chat, embeddings, health, rag
from app.routers import models as models_router
from app.services.llm_client import LLMClient

STATIC_DIR = Path(__file__).parent / "static"

log = logging.getLogger(__name__)


async def _warmup(client: LLMClient, settings) -> None:
    """Precalienta los modelos por defecto en background.

    Ollama carga el modelo en VRAM en el primer uso (y lo descarga al expirar
    OLLAMA_KEEP_ALIVE); ese cold-load son varios segundos que, sin warm-up,
    paga el primer mensaje del usuario. Aquí lo pagamos nosotros al arrancar:
    un chat de 1 token al modelo de chat por defecto y un embed mínimo al de
    embeddings (el RAG lo usa en la primera query). Cualquier fallo se loguea
    y se ignora: el warm-up nunca debe tumbar ni retrasar la API.
    """
    from app.models import ChatCompletionRequest, ChatMessage

    t0 = time.perf_counter()
    try:
        await client.chat(
            ChatCompletionRequest(
                model=settings.default_chat_model,
                messages=[ChatMessage(role="user", content="ping")],
                max_tokens=1,
                temperature=0.0,
            ),
            generation_name="warmup.chat",
        )
        log.info("warmup.chat_ok", extra={
            "model": settings.default_chat_model,
            "duration_s": round(time.perf_counter() - t0, 2),
        })
    except Exception as e:
        log.warning("warmup.chat_failed", extra={"model": settings.default_chat_model, "err": str(e)})
    t1 = time.perf_counter()
    try:
        await client.embed(settings.default_embed_model, ["warmup"], generation_name="warmup.embed")
        log.info("warmup.embed_ok", extra={
            "model": settings.default_embed_model,
            "duration_s": round(time.perf_counter() - t1, 2),
        })
    except Exception as e:
        log.warning("warmup.embed_failed", extra={"model": settings.default_embed_model, "err": str(e)})


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.api_log_level)

    # Langfuse (trazas LLM). Si está apagado, init_tracer devuelve un NoopTracer
    # y el resto de la app funciona idéntico sin emitir trazas.
    init_tracer(
        enabled=settings.langfuse_enabled,
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
        log_payloads=settings.langfuse_log_payloads,
    )

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
        embedder = Embedder(
            client=client,
            model=settings.default_embed_model,
            query_cache_size=settings.embed_query_cache_size,
        )
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

    warmup_task: asyncio.Task | None = None
    if settings.warmup_enabled:
        warmup_task = asyncio.create_task(_warmup(client, settings))

    log.info("api.startup", extra={"litellm_base_url": settings.litellm_base_url})
    try:
        yield
    finally:
        if warmup_task is not None and not warmup_task.done():
            warmup_task.cancel()
        if app.state.watcher:
            await app.state.watcher.stop()
        if app.state.rag and app.state.rag.get("store") is not None:
            await app.state.rag["store"].aclose()
        await client.aclose()
        # Flushea el buffer de Langfuse antes de morir para no perder trazas.
        shutdown_tracer()
        log.info("api.shutdown")


app = FastAPI(
    title="llm_ops API",
    description="Wrapper OpenAI-compatible delante de LiteLLM (Ollama/OpenAI/Anthropic), con RAG sobre Qdrant.",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(RequestContextMiddleware)


# Prometheus: instrumenta automáticamente latencia/RPS/status por endpoint y
# expone /metrics. Las métricas LLM-específicas (TTFT, tokens, tools, ...) las
# emitimos manualmente desde routers / LLMClient / AgentLoop usando el mismo
# REGISTRY global, así todo sale por el mismo endpoint.
def _setup_metrics(app: FastAPI) -> None:
    settings = get_settings()
    if not settings.metrics_enabled:
        return
    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        Instrumentator(
            should_group_status_codes=False,
            should_instrument_requests_inprogress=True,
            inprogress_labels=True,
            excluded_handlers=["/metrics", "/healthz", "/readyz"],
        ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
        log.info("metrics.ready")
    except Exception as e:
        log.warning("metrics.init_failed", extra={"err": str(e)})


_setup_metrics(app)


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
