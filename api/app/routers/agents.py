"""Endpoints del harness de agente."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.agents.loop import AgentLoop
from app.agents.registry import REGISTRY, get_tools
# Importar el módulo registra todas las tools
from app.agents.tools import rag_search  # noqa: F401
from app.models import AgentRunRequest, AgentToolInfo, AgentToolsResponse

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

    client = request.app.state.ollama
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

    async def event_stream():
        try:
            async for ev in loop.run(body.model, msgs, options=options):
                if await request.is_disconnected():
                    log.info("agent.client_disconnected")
                    return
                yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n".encode("utf-8")
        except Exception as e:
            log.exception("agent.stream_failed")
            err = {"type": "error", "message": f"{type(e).__name__}: {e}"}
            yield f"data: {json.dumps(err)}\n\n".encode("utf-8")
        finally:
            yield b"data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
