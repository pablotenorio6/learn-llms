"""Listado de modelos disponibles, en formato OpenAI."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.models import ModelInfo, ModelsResponse
from app.services.ollama_client import OllamaError

router = APIRouter(tags=["models"])


@router.get("/models", response_model=ModelsResponse)
async def list_models(request: Request) -> ModelsResponse:
    client = request.app.state.ollama
    try:
        ollama_models = await client.list_models()
    except OllamaError as e:
        raise HTTPException(status_code=502, detail=f"upstream error: {e}")

    items: list[ModelInfo] = []
    for m in ollama_models:
        # Ollama da `modified_at` en ISO; tiramos un epoch aproximado por compat
        # (no es crítico, OpenAI lo usa solo para ordenar)
        created = 0
        modified = m.get("modified_at")
        if isinstance(modified, str):
            try:
                from datetime import datetime
                created = int(datetime.fromisoformat(modified.replace("Z", "+00:00")).timestamp())
            except ValueError:
                created = 0
        items.append(
            ModelInfo(
                id=m.get("name", "unknown"),
                created=created,
                metadata={
                    "size_bytes": m.get("size"),
                    "digest": m.get("digest"),
                    "details": m.get("details"),
                },
            )
        )
    return ModelsResponse(data=items)
