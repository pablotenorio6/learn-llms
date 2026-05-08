"""Bucle del agente: model → tool_calls → execute → tool results → model → …

Uso:
    loop = AgentLoop(client, ctx, tools, max_iterations=10)
    async for event in loop.run(model, messages):
        # emit SSE
        ...

Eventos emitidos (cada uno es un dict serializable a JSON):
    {"type": "iteration", "n": int}
    {"type": "tool_call",   "id": str, "name": str, "arguments": dict}
    {"type": "tool_result", "id": str, "name": str, "result": Any, "duration_ms": float}
    {"type": "tool_error",  "id": str, "name": str, "error": str}
    {"type": "final",       "content": str, "iterations": int, "tool_calls": int}
    {"type": "limit_reached", "iterations": int, "tool_calls": int}
    {"type": "error",       "message": str}
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, AsyncIterator

from app.agents.registry import Tool, ToolArgsError, serialize_for_llm
from app.services.ollama_client import OllamaClient, OllamaError

log = logging.getLogger(__name__)


class AgentLoop:
    def __init__(
        self,
        client: OllamaClient,
        ctx: dict | None,
        tools: list[Tool],
        max_iterations: int = 10,
    ):
        self.client = client
        self.ctx = ctx or {}
        self.tools = tools
        self.tools_by_name: dict[str, Tool] = {t.name: t for t in tools}
        self.max_iterations = max_iterations

    async def run(
        self,
        model: str,
        messages: list[dict],
        options: dict | None = None,
    ) -> AsyncIterator[dict]:
        """Ejecuta el bucle. Itera hasta que el modelo deje de pedir tools, falle, o llegue al límite."""
        msgs = list(messages)
        tool_schemas = [t.schema for t in self.tools]
        total_tool_calls = 0

        for i in range(1, self.max_iterations + 1):
            yield {"type": "iteration", "n": i}

            try:
                resp = await self.client.chat_with_tools(
                    model=model,
                    messages=msgs,
                    tools=tool_schemas,
                    options=options,
                )
            except OllamaError as e:
                yield {"type": "error", "message": f"upstream error: {e}"}
                return

            msg = resp.get("message") or {}
            content = msg.get("content") or ""
            tcs = msg.get("tool_calls") or []

            # Caso 1: el modelo dice "ya está, esta es la respuesta"
            if not tcs:
                yield {
                    "type": "final",
                    "content": content,
                    "iterations": i,
                    "tool_calls": total_tool_calls,
                }
                return

            # Caso 2: hay tool_calls. Re-añadir el mensaje del asistente al historial
            # tal como vino (con tool_calls), y ejecutar cada una.
            msgs.append({
                "role": "assistant",
                "content": content,
                "tool_calls": tcs,
            })

            for tc in tcs:
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                # Ollama devuelve arguments como dict ya parseado; OpenAI manda string JSON.
                raw_args = fn.get("arguments") or {}
                if isinstance(raw_args, str):
                    try:
                        import json
                        raw_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        raw_args = {}

                call_id = uuid.uuid4().hex[:8]
                yield {"type": "tool_call", "id": call_id, "name": name, "arguments": raw_args}

                tool = self.tools_by_name.get(name)
                if not tool:
                    err = f"tool '{name}' no está en la allow-list"
                    yield {"type": "tool_error", "id": call_id, "name": name, "error": err}
                    msgs.append({"role": "tool", "name": name, "content": serialize_for_llm({"error": err})})
                    total_tool_calls += 1
                    continue

                start = time.perf_counter()
                try:
                    result = await tool.call(self.ctx, raw_args)
                    duration_ms = round((time.perf_counter() - start) * 1000, 2)
                    yield {
                        "type": "tool_result",
                        "id": call_id,
                        "name": name,
                        "result": result,
                        "duration_ms": duration_ms,
                    }
                    msgs.append({
                        "role": "tool",
                        "name": name,
                        "content": serialize_for_llm(result),
                    })
                except ToolArgsError as e:
                    err = f"args inválidos: {e}"
                    yield {"type": "tool_error", "id": call_id, "name": name, "error": err}
                    msgs.append({"role": "tool", "name": name, "content": serialize_for_llm({"error": err})})
                except Exception as e:
                    log.exception("agent.tool_failed", extra={"tool": name})
                    err = f"{type(e).__name__}: {e}"
                    yield {"type": "tool_error", "id": call_id, "name": name, "error": err}
                    msgs.append({"role": "tool", "name": name, "content": serialize_for_llm({"error": err})})
                total_tool_calls += 1

        # Llegamos al límite sin que el modelo cierre
        yield {
            "type": "limit_reached",
            "iterations": self.max_iterations,
            "tool_calls": total_tool_calls,
        }
