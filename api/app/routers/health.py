"""Liveness y readiness probes."""

from __future__ import annotations

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from app.services.llm_client import LLMError

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz():
    """Liveness: el proceso responde. Sin dependencias externas."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request):
    """Readiness: LiteLLM proxy es alcanzable y devuelve los aliases configurados."""
    client = request.app.state.llm
    try:
        models = await client.list_models()
        return {"status": "ready", "models_available": len(models)}
    except LLMError as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "reason": str(e)},
        )
