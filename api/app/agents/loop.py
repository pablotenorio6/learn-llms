"""Bucle del agente con streaming.

Eventos emitidos:
    {"type": "iteration", "n": int}
    {"type": "content_delta", "content": str}        ← tokens streamed del asistente
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
from app.observability import (
    agent_iterations_total,
    agent_tool_calls_total,
    agent_tool_duration,
    current_observation_id,
    get_tracer,
)
from app.services.llm_client import LLMClient, LLMError

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
- Aritmética trivial y de cabeza (p.ej. 2+2) y lógica simple. Para cálculo NO trivial usa `calculator`.

TOOLS DE CÓMPUTO Y SISTEMA:
- `calculator`: aritmética no trivial, potencias, raíces, logaritmos, trigonometría. No te fíes del cálculo mental en operaciones de varios dígitos: si el error importa, calcula.
- `datetime_now`: fecha y hora actuales. Úsala SIEMPRE que la respuesta dependa del momento presente (qué día es hoy, cuántos días faltan para X, edades, vencimientos) — no conoces la fecha actual por tu cuenta.
- `python_exec`: ejecuta código Python para cálculos multi-paso, manipular datos o texto, o simulaciones. Imprime los resultados con print(); no hay acceso a internet.
- `fs_list` / `fs_read` / `fs_write`: workspace de archivos del agente (sandbox). Úsalas para guardar resultados intermedios y recuperarlos después.

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
        client: LLMClient,
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
        tracer = get_tracer()

        for i in range(1, self.max_iterations + 1):
            yield {"type": "iteration", "n": i}
            agent_iterations_total.labels(model=model).inc()

            # Span por iteración: bajo él colgarán la generation de LiteLLM y
            # los spans de cada tool ejecutada en esta vuelta.
            iter_handle = tracer.start_span(
                f"agent.iteration.{i}",
                metadata={"iteration": i, "model": model},
            )
            iter_token = current_observation_id.set(iter_handle.id) if iter_handle.id else None

            accumulated_content = ""
            # tool_calls llegan en streaming como deltas con `index` estable y
            # `function.arguments` que se va concatenando chunk a chunk.
            tool_calls_accum: dict[int, dict[str, Any]] = {}
            finish_reason: str | None = None
            iter_status: str = "ok"
            iter_status_message: str | None = None

            try:
                try:
                    async for chunk in self.client.chat_with_tools_stream(
                        model=model,
                        messages=msgs,
                        tools=tool_schemas,
                        options=options,
                        generation_name=f"agent.iter{i}.llm",
                    ):
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        choice = choices[0]
                        delta = choice.get("delta") or {}

                        piece = delta.get("content") or ""
                        if piece:
                            accumulated_content += piece
                            yield {"type": "content_delta", "content": piece}

                        for tcd in delta.get("tool_calls") or []:
                            idx = tcd.get("index", 0)
                            slot = tool_calls_accum.setdefault(
                                idx,
                                {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                            )
                            if tcd.get("id"):
                                slot["id"] = tcd["id"]
                            if tcd.get("type"):
                                slot["type"] = tcd["type"]
                            fn = tcd.get("function") or {}
                            if fn.get("name"):
                                slot["function"]["name"] += fn["name"]
                            if fn.get("arguments"):
                                slot["function"]["arguments"] += fn["arguments"]

                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
                except LLMError as e:
                    iter_status = "error"
                    iter_status_message = f"upstream error: {e}"
                    yield {"type": "error", "message": iter_status_message}
                    return

                collected_tool_calls = [tool_calls_accum[k] for k in sorted(tool_calls_accum.keys())]

                # Si no hay tool_calls válidas, es respuesta final.
                # Importante: NO miramos finish_reason — LiteLLM con Ollama emite
                # finish_reason="stop" aunque haya tool_calls en el delta, mientras
                # que OpenAI puro emite "tool_calls". Confiamos en la presencia del
                # tool_call para decidir ramificar.
                if not collected_tool_calls:
                    yield {
                        "type": "final",
                        "content": accumulated_content,
                        "iterations": i,
                        "tool_calls": total_tool_calls,
                    }
                    return

                # Si la id viene vacía (algunos proveedores la omiten en streaming),
                # generamos una local para poder referenciarla en role=tool.
                for tc in collected_tool_calls:
                    if not tc.get("id"):
                        tc["id"] = f"call_{uuid.uuid4().hex[:16]}"

                # Mensaje del asistente con los tool_calls completos.
                msgs.append({
                    "role": "assistant",
                    "content": accumulated_content or None,
                    "tool_calls": collected_tool_calls,
                })

                for tc in collected_tool_calls:
                    call_id = tc["id"]
                    fn = tc.get("function") or {}
                    name = fn.get("name") or ""
                    raw_args: Any = fn.get("arguments") or "{}"
                    if isinstance(raw_args, str):
                        try:
                            raw_args = json.loads(raw_args) if raw_args.strip() else {}
                        except json.JSONDecodeError:
                            raw_args = {}

                    yield {"type": "tool_call", "id": call_id, "name": name, "arguments": raw_args}
                    total_tool_calls += 1

                    tool = self.tools_by_name.get(name)
                    if not tool:
                        err = f"tool '{name}' no está en la allow-list"
                        agent_tool_calls_total.labels(tool=name or "unknown", outcome="unknown").inc()
                        yield {"type": "tool_error", "id": call_id, "name": name, "error": err}
                        msgs.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": serialize_for_llm({"error": err}),
                        })
                        continue

                    # Span por tool: cuelga del span de iteración (current_observation_id).
                    tool_span = tracer.start_span(
                        f"tool.{name}",
                        input=raw_args if tracer.log_payloads else None,
                        metadata={"tool_call_id": call_id},
                    )
                    start = time.perf_counter()
                    outcome = "ok"
                    try:
                        result = await tool.call(self.ctx, raw_args)
                        duration_s = time.perf_counter() - start
                        agent_tool_duration.labels(tool=name).observe(duration_s)
                        if tracer.log_payloads:
                            tool_span.update(output=result)
                        yield {
                            "type": "tool_result",
                            "id": call_id,
                            "name": name,
                            "result": result,
                            "duration_ms": round(duration_s * 1000, 2),
                        }
                        msgs.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": serialize_for_llm(result),
                        })
                    except ToolArgsError as e:
                        outcome = "bad_args"
                        err = f"args inválidos: {e}"
                        tool_span.update(level="ERROR", status_message=err)
                        yield {"type": "tool_error", "id": call_id, "name": name, "error": err}
                        msgs.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": serialize_for_llm({"error": err}),
                        })
                    except Exception as e:
                        outcome = "error"
                        log.exception("agent.tool_failed", extra={"tool": name})
                        err = f"{type(e).__name__}: {e}"
                        tool_span.update(level="ERROR", status_message=err)
                        yield {"type": "tool_error", "id": call_id, "name": name, "error": err}
                        msgs.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": serialize_for_llm({"error": err}),
                        })
                    finally:
                        tool_span.end()
                        agent_tool_calls_total.labels(tool=name, outcome=outcome).inc()
            finally:
                if iter_status == "error" and iter_status_message:
                    iter_handle.update(level="ERROR", status_message=iter_status_message)
                iter_handle.end()
                if iter_token is not None:
                    current_observation_id.reset(iter_token)

        yield {
            "type": "limit_reached",
            "iterations": self.max_iterations,
            "tool_calls": total_tool_calls,
        }
