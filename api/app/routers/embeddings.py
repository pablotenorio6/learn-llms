"""POST /v1/embeddings (compatible OpenAI)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.models import EmbeddingItem, EmbeddingRequest, EmbeddingResponse, Usage
from app.services.ollama_client import OllamaError

router = APIRouter(tags=["embeddings"])


@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(req: EmbeddingRequest, request: Request) -> EmbeddingResponse:
    if req.encoding_format != "float":
        raise HTTPException(
            status_code=400,
            detail="only encoding_format='float' is supported",
        )

    inputs = [req.input] if isinstance(req.input, str) else list(req.input)
    if not inputs:
        raise HTTPException(status_code=400, detail="`input` cannot be empty")

    client = request.app.state.ollama
    try:
        result = await client.embed(req.model, inputs)
    except OllamaError as e:
        raise HTTPException(status_code=502, detail=f"upstream error: {e}")

    embeddings = result.get("embeddings") or []
    if len(embeddings) != len(inputs):
        raise HTTPException(
            status_code=502,
            detail=f"upstream returned {len(embeddings)} embeddings for {len(inputs)} inputs",
        )

    items = [
        EmbeddingItem(index=i, embedding=list(vec))
        for i, vec in enumerate(embeddings)
    ]

    prompt_tokens = int(result.get("prompt_eval_count") or 0)
    return EmbeddingResponse(
        data=items,
        model=req.model,
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=0,
            total_tokens=prompt_tokens,
        ),
    )
