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
from app.observability import current_trace_id, get_tracer
from app.services.llm_client import LLMClient, LLMError

log = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


def _new_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


@router.post("/chat/completions")
async def create_chat_completion(body: ChatCompletionRequest, request: Request):
    if not body.messages:
        raise HTTPException(status_code=400, detail="`messages` cannot be empty")

    client: LLMClient = request.app.state.llm
    tracer = get_tracer()

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

    # Non-streaming. LiteLLM ya devuelve un objeto OpenAI estándar; lo
    # re-empaquetamos en nuestros schemas (que añaden id/object/etc. propios).
    with tracer.trace(
        "chat.completions",
        input=[m.model_dump(exclude_none=True) for m in body.messages] if tracer.log_payloads else None,
        metadata={"model": body.model, "stream": False},
    ) as trace_handle:
        try:
            result = await client.chat(body, generation_name="chat")
        except LLMError as e:
            trace_handle.update(level="ERROR", status_message=str(e))
            raise HTTPException(status_code=502, detail=f"upstream error: {e}")

        choices_raw = result.get("choices") or []
        if not choices_raw:
            raise HTTPException(status_code=502, detail="upstream returned no choices")
        first = choices_raw[0]
        msg = first.get("message") or {}
        content = msg.get("content") or ""
        finish = first.get("finish_reason") or "stop"

        usage_raw = result.get("usage") or {}
        if tracer.log_payloads:
            trace_handle.update(output=content)
        return ChatCompletionResponse(
            id=_new_id(),
            created=int(time.time()),
            model=result.get("model") or body.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason=finish,
                )
            ],
            usage=Usage(
                prompt_tokens=int(usage_raw.get("prompt_tokens") or 0),
                completion_tokens=int(usage_raw.get("completion_tokens") or 0),
                total_tokens=int(usage_raw.get("total_tokens") or 0),
            ),
        )


async def _stream_chunks(
    client: LLMClient,
    body: ChatCompletionRequest,
    request: Request,
) -> AsyncIterator[bytes]:
    """Generador SSE en formato OpenAI.

    Re-emitimos chunks como ChatCompletionChunk de nuestro modelo para
    mantener un id/object estables (LiteLLM ya manda objetos compatibles,
    pero los normalizamos por si añadimos fields nuestros).
    """
    completion_id = _new_id()
    created = int(time.time())
    tracer = get_tracer()

    def _sse(chunk: ChatCompletionChunk) -> bytes:
        return f"data: {chunk.model_dump_json()}\n\n".encode("utf-8")

    # Abrimos trace manualmente (no con context manager) porque el handler
    # devuelve antes de que el generador se consuma; queremos que el trace
    # viva todo el streaming.
    trace_handle = tracer.start_trace(
        "chat.completions",
        input=[m.model_dump(exclude_none=True) for m in body.messages] if tracer.log_payloads else None,
        metadata={"model": body.model, "stream": True},
    )
    token = current_trace_id.set(trace_handle.id)

    accumulated_output = ""
    error_message: str | None = None
    try:
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
        # is_disconnected() hace un receive() del ASGI por llamada; a decenas
        # de tokens/s es overhead puro. Lo comprobamos como mucho cada 250 ms:
        # la cancelación sigue siendo casi inmediata y el hot path por token
        # queda limpio.
        last_disconnect_check = 0.0

        try:
            async for raw in client.chat_stream(body, generation_name="chat.stream"):
                now = time.monotonic()
                if now - last_disconnect_check >= 0.25:
                    last_disconnect_check = now
                    if await request.is_disconnected():
                        log.info("chat.client_disconnected", extra={"id": completion_id})
                        return

                choices = raw.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                piece = delta.get("content") or ""
                if piece:
                    accumulated_output += piece
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

                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                    break
        except LLMError as e:
            error_message = str(e)
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
    finally:
        if error_message:
            trace_handle.update(level="ERROR", status_message=error_message)
        elif tracer.log_payloads and accumulated_output:
            trace_handle.update(output=accumulated_output)
        trace_handle.end()
        current_trace_id.reset(token)
