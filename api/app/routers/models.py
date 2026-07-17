"""Listado de modelos disponibles, en formato OpenAI.

La fuente de verdad ahora son los aliases declarados en litellm-config.yaml,
expuestos por el proxy en GET /v1/models.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request

from app.config import get_settings
from app.models import ModelInfo, ModelsResponse
from app.services.llm_client import LLMError

router = APIRouter(tags=["models"])

# Cache TTL del listado: la UI lo pide en cada carga de página y los aliases
# solo cambian al editar litellm-config.yaml + reiniciar el proxy. Con el TTL
# nos ahorramos el roundtrip al proxy en el camino crítico de carga de la UI.
_cache: dict = {"at": 0.0, "resp": None}


@router.get("/models", response_model=ModelsResponse)
async def list_models(request: Request) -> ModelsResponse:
    ttl = get_settings().models_cache_ttl_s
    if _cache["resp"] is not None and ttl > 0 and time.monotonic() - _cache["at"] < ttl:
        return _cache["resp"]

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
    resp = ModelsResponse(data=items)
    _cache["at"] = time.monotonic()
    _cache["resp"] = resp
    return resp
