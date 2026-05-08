"""Liveness y readiness probes."""

from __future__ import annotations

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from app.services.ollama_client import OllamaError

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz():
    """Liveness: el proceso responde. Sin dependencias externas."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request):
    """Readiness: Ollama es alcanzable y responde."""
    client = request.app.state.ollama
    try:
        models = await client.list_models()
        return {"status": "ready", "models_loaded": len(models)}
    except OllamaError as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "reason": str(e)},
        )
