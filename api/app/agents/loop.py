"""Bucle del agente con streaming.

Eventos emitidos:
    {"type": "iteration", "n": int}
    {"type": "content_delta", "content": str}        ← nuevo: tokens streamed del asistente
    {"type": "tool_call",   "id": str, "name": str, "arguments": dict}
    {"type": "tool_result", "id": str, "name": str, "result": Any, "duration_ms": float}
    {"type": "tool_error",  "id": str, "name": str, "error": str}
    {"type": "final",       "content": str, "iterations": int, "tool_calls": int}
    {"type": "limit_reached", "iterations": int, "tool_calls": int}
    {"type": "error",       "message": str}
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, AsyncIterator

from app.agents.registry import Tool, ToolArgsError, serialize_for_llm
from app.services.ollama_client import OllamaClient, OllamaError

log = logging.getLogger(__name__)


DEFAULT_AGENT_SYSTEM = """\
Eres un asistente con acceso a tools que extienden tu conocimiento. Tu objetivo es responder al usuario con la mejor respuesta posible.

DECISIÓN INTERNA (no la comuniques al usuario):
Para cada mensaje, decides internamente si necesitas alguna tool. Esta deliberación es invisible para el usuario.

CUÁNDO LLAMAR UNA TOOL:
- La pregunta hace referencia (explícita o implícita) a los documentos, archivos, notas, base de datos, RAG, base de conocimiento o índice del usuario → usa `rag_search`.
- La pregunta requiere información actual, eventos recientes, noticias, precios, documentación de software, hechos verificables que pueden haber cambiado, o cualquier dato externo que no podrías conocer por entrenamiento → usa `web_search`.
- Ante la duda razonable, llámala — si vuelve vacía, lo reconocerás brevemente.

FLUJO OBLIGATORIO `web_search` → `http_fetch`:
Los `snippet` que devuelve `web_search` son fragmentos de 1-2 frases, NO la página completa. Bastan solo para hechos puntuales (una fecha, un nombre, una capital). Tras `web_search`, DEBES encadenar con `http_fetch` sobre la URL del resultado más relevante SI la pregunta del usuario implica cualquiera de estos casos:
- Pide explicación, detalle, contexto, "cómo funciona", "por qué", o un resumen sustancial.
- Pide instrucciones paso a paso, un tutorial, código o ejemplos.
- Pide cifras específicas, citas textuales, o comparar varias fuentes.
- Pide leer/resumir un artículo, una doc, una noticia, un post o cualquier página concreta.
- El snippet menciona el tema pero no contiene la respuesta concreta.

Solo te puedes saltar `http_fetch` cuando la respuesta cabe literalmente en el snippet (ej. "¿quién ganó las elecciones X en 2024?" y el snippet dice "Ganó Fulano con el 52%"). En cualquier otro caso, encadena `http_fetch` ANTES de redactar la respuesta final. No anuncies "voy a abrir la página" — simplemente llámala.

Solo usa `http_fetch` con URLs obtenidas de `web_search` o que el usuario haya dado.

CUÁNDO RESPONDER DIRECTAMENTE SIN TOOLS:
- Conocimiento general claramente estable y público (geografía elemental, ciencia básica, definiciones comunes).
- Saludos, conversación trivial, opiniones, escritura creativa, traducción.
- Matemáticas, cálculo, lógica.

DISTINCIÓN IMPORTANTE:
- "Mis documentos", "mi PDF", "mis notas", "mi base de datos/RAG" → `rag_search` (NUNCA `web_search`).
- "Busca en internet", "últimas noticias de X", "qué dice la doc oficial de Y", "qué pasó con Z hoy" → `web_search`.

CÓMO REDACTAR TU RESPUESTA AL USUARIO:
- Responde DIRECTAMENTE al contenido de la pregunta. Nada más.
- NO narres tu decisión sobre tools. Nunca digas "voy a llamar X", "no llamaré X", "buscaré en mi base de datos", "consultaré tus documentos", "no necesito buscar", o variantes.
- NO te disculpes por límites que no tienes. Tienes acceso a tus tools — no digas "no tengo acceso a tu base de datos" porque sí lo tienes.
- NO confundas "documentos del usuario" con "datos personales sensibles": el contenido del RAG es información que el usuario te ha dado para que la consultes, no PII que debas proteger.
- Si una tool devuelve resultados útiles, intégralos naturalmente y cita la fuente cuando sea relevante.
- Si una tool devuelve resultados vacíos o irrelevantes, di brevemente que no encontraste eso en los documentos y responde con tu conocimiento general.
- No llames la misma tool con los mismos argumentos dos veces seguidas."""


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
        inject_default_system: bool = True,
    ) -> AsyncIterator[dict]:
        msgs = list(messages)
        if inject_default_system:
            has_system = bool(msgs) and msgs[0].get("role") == "system"
            if has_system:
                msgs[0] = {
                    "role": "system",
                    "content": DEFAULT_AGENT_SYSTEM + "\n\n---\n\n" + msgs[0].get("content", ""),
                }
            else:
                msgs.insert(0, {"role": "system", "content": DEFAULT_AGENT_SYSTEM})

        tool_schemas = [t.schema for t in self.tools]
        total_tool_calls = 0

        for i in range(1, self.max_iterations + 1):
            yield {"type": "iteration", "n": i}

            accumulated_content = ""
            collected_tool_calls: list[dict] = []

            try:
                async for chunk in self.client.chat_with_tools_stream(
                    model=model,
                    messages=msgs,
                    tools=tool_schemas,
                    options=options,
                ):
                    msg = chunk.get("message") or {}
                    piece = msg.get("content") or ""
                    if piece:
                        accumulated_content += piece
                        yield {"type": "content_delta", "content": piece}
                    tcs = msg.get("tool_calls") or []
                    if tcs:
                        collected_tool_calls.extend(tcs)
                    # done=true llega en el último chunk; el bucle terminará al cerrarse el stream
            except OllamaError as e:
                yield {"type": "error", "message": f"upstream error: {e}"}
                return

            if not collected_tool_calls:
                yield {
                    "type": "final",
                    "content": accumulated_content,
                    "iterations": i,
                    "tool_calls": total_tool_calls,
                }
                return

            # Ramas con tool_calls: añadir el msg del asistente y ejecutar
            msgs.append({
                "role": "assistant",
                "content": accumulated_content,
                "tool_calls": collected_tool_calls,
            })

            for tc in collected_tool_calls:
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                raw_args = fn.get("arguments") or {}
                if isinstance(raw_args, str):
                    try:
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

        yield {
            "type": "limit_reached",
            "iterations": self.max_iterations,
            "tool_calls": total_tool_calls,
        }
