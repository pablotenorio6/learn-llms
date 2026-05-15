"""Listado de modelos disponibles, en formato OpenAI.

La fuente de verdad ahora son los aliases declarados en litellm-config.yaml,
expuestos por el proxy en GET /v1/models.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.models import ModelInfo, ModelsResponse
from app.services.llm_client import LLMError

router = APIRouter(tags=["models"])


@router.get("/models", response_model=ModelsResponse)
async def list_models(request: Request) -> ModelsResponse:
    client = request.app.state.llm
    try:
        upstream = await client.list_models()
    except LLMError as e:
        raise HTTPException(status_code=502, detail=f"upstream error: {e}")

    items: list[ModelInfo] = []
    for m in upstream:
        created = int(m.get("created") or 0)
        owned_by = m.get("owned_by") or "litellm"
        items.append(
            ModelInfo(
                id=m.get("id", "unknown"),
                created=created,
                owned_by=owned_by,
                metadata={k: v for k, v in m.items() if k not in {"id", "created", "object", "owned_by"}} or None,
            )
        )
    return ModelsResponse(data=items)
