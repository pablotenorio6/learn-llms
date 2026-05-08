"""POST /v1/chat/completions (compatible OpenAI), con y sin streaming."""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.models import (
    ChatChoice,
    ChatChoiceDelta,
    ChatChunkChoice,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Usage,
)
from app.services.ollama_client import OllamaClient, OllamaError

log = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


def _new_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


@router.post("/chat/completions")
async def create_chat_completion(body: ChatCompletionRequest, request: Request):
    if not body.messages:
        raise HTTPException(status_code=400, detail="`messages` cannot be empty")

    client: OllamaClient = request.app.state.ollama

    if body.stream:
        return StreamingResponse(
            _stream_chunks(client, body, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # nginx
                "Connection": "keep-alive",
            },
        )

    # Non-streaming
    try:
        result = await client.chat(body)
    except OllamaError as e:
        raise HTTPException(status_code=502, detail=f"upstream error: {e}")

    msg = result.get("message") or {}
    content = msg.get("content", "")
    finish = "length" if result.get("done_reason") == "length" else "stop"

    return ChatCompletionResponse(
        id=_new_id(),
        created=int(time.time()),
        model=body.model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=content),
                finish_reason=finish,
            )
        ],
        usage=Usage(
            prompt_tokens=int(result.get("prompt_eval_count") or 0),
            completion_tokens=int(result.get("eval_count") or 0),
            total_tokens=int(result.get("prompt_eval_count") or 0)
            + int(result.get("eval_count") or 0),
        ),
    )


async def _stream_chunks(
    client: OllamaClient,
    body: ChatCompletionRequest,
    request: Request,
) -> AsyncIterator[bytes]:
    """Generador SSE en formato OpenAI."""
    completion_id = _new_id()
    created = int(time.time())

    def _sse(chunk: ChatCompletionChunk) -> bytes:
        return f"data: {chunk.model_dump_json()}\n\n".encode("utf-8")

    # Primer chunk con role (como hace OpenAI)
    yield _sse(
        ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=body.model,
            choices=[ChatChunkChoice(index=0, delta=ChatChoiceDelta(role="assistant"))],
        )
    )

    finish_reason: str | None = None

    try:
        async for raw in client.chat_stream(body):
            # Si el cliente cerró la conexión, abortamos
            if await request.is_disconnected():
                log.info("chat.client_disconnected", extra={"id": completion_id})
                return

            piece = (raw.get("message") or {}).get("content") or ""
            if piece:
                yield _sse(
                    ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=body.model,
                        choices=[
                            ChatChunkChoice(index=0, delta=ChatChoiceDelta(content=piece))
                        ],
                    )
                )

            if raw.get("done"):
                finish_reason = "length" if raw.get("done_reason") == "length" else "stop"
                break
    except OllamaError as e:
        # En medio del stream solo podemos notificar como evento de error
        err = {"error": {"message": str(e), "type": "api_error", "code": 502}}
        yield f"data: {json.dumps(err)}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"
        return

    # Chunk final con finish_reason
    yield _sse(
        ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=body.model,
            choices=[
                ChatChunkChoice(
                    index=0,
                    delta=ChatChoiceDelta(),
                    finish_reason=finish_reason or "stop",
                )
            ],
        )
    )
    yield b"data: [DONE]\n\n"
