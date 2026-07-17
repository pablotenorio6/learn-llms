"""Endpoints del harness de agente."""

from __future__ import annotations

import json
import logging
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.agents.loop import AgentLoop
from app.agents.registry import REGISTRY, get_tools
# Importar el paquete registra todas las tools (rag_search, web_search, http_fetch, ...)
from app.agents import tools as _tools  # noqa: F401
from app.models import AgentRunRequest, AgentToolInfo, AgentToolsResponse
from app.observability import current_trace_id, get_tracer

log = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("/tools", response_model=AgentToolsResponse)
async def list_tools():
    """Lista las tools registradas."""
    return AgentToolsResponse(data=[
        AgentToolInfo(
            name=t.name,
            description=t.description,
            parameters=t.schema["function"]["parameters"],
        )
        for t in REGISTRY.values()
    ])


@router.post("/run")
async def run_agent(body: AgentRunRequest, request: Request):
    """Corre el bucle del agente y streamea eventos por SSE.

    Eventos: iteration, tool_call, tool_result, tool_error, final, limit_reached, error, [DONE].
    """
    if not body.messages:
        raise HTTPException(status_code=400, detail="`messages` cannot be empty")

    # Resolver tools
    if body.tools_allowed is not None:
        unknown = [t for t in body.tools_allowed if t not in REGISTRY]
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"tools desconocidas: {unknown}. disponibles: {sorted(REGISTRY.keys())}",
            )
        tools = get_tools(allow=body.tools_allowed)
    else:
        tools = get_tools()

    if not tools:
        raise HTTPException(status_code=400, detail="no hay tools disponibles")

    client = request.app.state.llm
    ctx = {
        "rag": getattr(request.app.state, "rag", None),
    }

    options = {}
    if body.temperature is not None:
        options["temperature"] = body.temperature

    loop = AgentLoop(
        client=client,
        ctx=ctx,
        tools=tools,
        max_iterations=body.max_iterations,
    )

    msgs = [m.model_dump(exclude_none=True) for m in body.messages]
    tracer = get_tracer()
    tool_names = [t.name for t in tools]

    async def event_stream():
        # Abrimos trace manualmente para que viva hasta que el generador acabe.
        trace_handle = tracer.start_trace(
            "agent.run",
            input=msgs if tracer.log_payloads else None,
            metadata={
                "model": body.model,
                "max_iterations": body.max_iterations,
                "tools_available": tool_names,
            },
            tags=["agent"],
        )
        token = current_trace_id.set(trace_handle.id)
        final_content: str | None = None
        final_error: str | None = None
        iter_count = 0
        tool_count = 0
        # Comprobar desconexión como mucho cada 250 ms: is_disconnected() hace
        # un receive() del ASGI por llamada y los content_delta llegan a
        # decenas por segundo.
        last_disconnect_check = 0.0
        try:
            try:
                async for ev in loop.run(body.model, msgs, options=options):
                    now = time.monotonic()
                    if now - last_disconnect_check >= 0.25:
                        last_disconnect_check = now
                        if await request.is_disconnected():
                            log.info("agent.client_disconnected")
                            return
                    if ev.get("type") == "final":
                        final_content = ev.get("content")
                        iter_count = int(ev.get("iterations") or 0)
                        tool_count = int(ev.get("tool_calls") or 0)
                    elif ev.get("type") == "limit_reached":
                        iter_count = int(ev.get("iterations") or 0)
                        tool_count = int(ev.get("tool_calls") or 0)
                        final_error = "limit_reached"
                    elif ev.get("type") == "error":
                        final_error = ev.get("message") or "error"
                    yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n".encode("utf-8")
            except Exception as e:
                log.exception("agent.stream_failed")
                final_error = f"{type(e).__name__}: {e}"
                err = {"type": "error", "message": final_error}
                yield f"data: {json.dumps(err)}\n\n".encode("utf-8")
            finally:
                yield b"data: [DONE]\n\n"
        finally:
            if final_error:
                trace_handle.update(level="ERROR", status_message=final_error,
                                    metadata={"iterations": iter_count, "tool_calls": tool_count})
            else:
                trace_handle.update(
                    output=final_content if tracer.log_payloads else None,
                    metadata={"iterations": iter_count, "tool_calls": tool_count},
                )
            trace_handle.end()
            current_trace_id.reset(token)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
