"""Cliente async contra LiteLLM proxy.

La API ya no habla directamente con Ollama / OpenAI / Anthropic. Habla con
LiteLLM en formato OpenAI puro, y LiteLLM enruta al backend correcto según
el alias del modelo declarado en litellm-config.yaml.

Usamos el SDK oficial de OpenAI (AsyncOpenAI) apuntado a la base_url del
proxy. Esto nos da streaming SSE bien parseado y typed-objects en vez de
dicts a pelo.

Observabilidad: cada método mide latencia + TTFT + tokens y los emite a
Prometheus, y forwardea metadata Langfuse (si hay trace activo) al proxy
mediante `extra_body={"metadata": ...}` para que LiteLLM cree la generation
bajo el trace correcto.
"""

from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator

from openai import APIError, APIStatusError, AsyncOpenAI

from app.models import ChatCompletionRequest, ChatMessage
from app.observability import (
    get_tracer,
    llm_active_requests,
    llm_request_duration,
    llm_tokens_total,
    llm_ttft,
)

log = logging.getLogger(__name__)


class LLMError(Exception):
    """Cualquier fallo hablando con LiteLLM (HTTP, parseo, upstream)."""


class LLMClient:
    """Cliente unificado contra LiteLLM proxy.

    Métodos:
      - chat / chat_stream: chat completions con/sin streaming.
      - chat_with_tools / chat_with_tools_stream: ídem con tool calling.
      - embed: embeddings.
      - list_models: alias declarados en el proxy.
    """

    def __init__(self, base_url: str, master_key: str, timeout: int = 600):
        self._client = AsyncOpenAI(
            base_url=base_url.rstrip("/") + "/v1",
            api_key=master_key,
            timeout=timeout,
            max_retries=0,  # LiteLLM ya reintenta del lado del proxy
        )
        self.base_url = base_url
        self.timeout = timeout

    async def aclose(self) -> None:
        await self._client.close()

    # ---- models ----

    async def list_models(self) -> list[dict[str, Any]]:
        try:
            page = await self._client.models.list()
        except APIError as e:
            raise LLMError(f"network error: {e}") from e
        return [m.model_dump() for m in page.data]

    # ---- chat (sin tools) ----

    async def chat(
        self,
        req: ChatCompletionRequest,
        *,
        generation_name: str = "chat",
    ) -> dict[str, Any]:
        """Chat completions no-streaming. Devuelve el dict (model_dump) crudo."""
        payload = self._build_chat_payload(req, stream=False)
        self._attach_langfuse(payload, generation_name)
        endpoint = "chat"
        status = "error"
        t0 = time.perf_counter()
        llm_active_requests.labels(endpoint=endpoint).inc()
        try:
            try:
                resp = await self._client.chat.completions.create(**payload)
            except APIStatusError as e:
                raise LLMError(f"upstream {e.status_code}: {e.message}") from e
            except APIError as e:
                raise LLMError(f"network error: {e}") from e
            data = resp.model_dump()
            status = "ok"
            _record_tokens(req.model, data.get("usage"))
            return data
        finally:
            llm_active_requests.labels(endpoint=endpoint).dec()
            llm_request_duration.labels(
                model=req.model, endpoint=endpoint, stream="false", status=status
            ).observe(time.perf_counter() - t0)

    async def chat_stream(
        self,
        req: ChatCompletionRequest,
        *,
        generation_name: str = "chat.stream",
    ) -> AsyncIterator[dict[str, Any]]:
        """Chat completions con streaming. Yields chunk.model_dump() por chunk."""
        payload = self._build_chat_payload(req, stream=True)
        self._attach_langfuse(payload, generation_name)
        endpoint = "chat"
        status = "error"
        t0 = time.perf_counter()
        first_token_logged = False
        llm_active_requests.labels(endpoint=endpoint).inc()
        try:
            try:
                stream = await self._client.chat.completions.create(**payload)
                async for chunk in stream:
                    data = chunk.model_dump()
                    if not first_token_logged and _chunk_has_content(data):
                        llm_ttft.labels(model=req.model, endpoint=endpoint).observe(
                            time.perf_counter() - t0
                        )
                        first_token_logged = True
                    usage = data.get("usage")
                    if usage:
                        _record_tokens(req.model, usage)
                    yield data
                status = "ok"
            except APIStatusError as e:
                raise LLMError(f"upstream {e.status_code}: {e.message}") from e
            except APIError as e:
                raise LLMError(f"stream error: {e}") from e
        finally:
            llm_active_requests.labels(endpoint=endpoint).dec()
            llm_request_duration.labels(
                model=req.model, endpoint=endpoint, stream="true", status=status
            ).observe(time.perf_counter() - t0)

    # ---- chat con tools ----

    async def chat_with_tools(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
        *,
        generation_name: str = "agent.llm",
    ) -> dict[str, Any]:
        """Chat con tool calling, no-streaming."""
        payload = self._build_tools_payload(model, messages, tools, options, stream=False)
        self._attach_langfuse(payload, generation_name)
        endpoint = "agent"
        status = "error"
        t0 = time.perf_counter()
        llm_active_requests.labels(endpoint=endpoint).inc()
        try:
            try:
                resp = await self._client.chat.completions.create(**payload)
            except APIStatusError as e:
                raise LLMError(f"upstream {e.status_code}: {e.message}") from e
            except APIError as e:
                raise LLMError(f"network error: {e}") from e
            data = resp.model_dump()
            status = "ok"
            _record_tokens(model, data.get("usage"))
            return data
        finally:
            llm_active_requests.labels(endpoint=endpoint).dec()
            llm_request_duration.labels(
                model=model, endpoint=endpoint, stream="false", status=status
            ).observe(time.perf_counter() - t0)

    async def chat_with_tools_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
        *,
        generation_name: str = "agent.llm.stream",
    ) -> AsyncIterator[dict[str, Any]]:
        """Chat con tool calling, streaming. Yields chunks en formato OpenAI."""
        payload = self._build_tools_payload(model, messages, tools, options, stream=True)
        self._attach_langfuse(payload, generation_name)
        endpoint = "agent"
        status = "error"
        t0 = time.perf_counter()
        first_token_logged = False
        llm_active_requests.labels(endpoint=endpoint).inc()
        try:
            try:
                stream = await self._client.chat.completions.create(**payload)
                async for chunk in stream:
                    data = chunk.model_dump()
                    if not first_token_logged and _chunk_has_content(data):
                        llm_ttft.labels(model=model, endpoint=endpoint).observe(
                            time.perf_counter() - t0
                        )
                        first_token_logged = True
                    usage = data.get("usage")
                    if usage:
                        _record_tokens(model, usage)
                    yield data
                status = "ok"
            except APIStatusError as e:
                raise LLMError(f"upstream {e.status_code}: {e.message}") from e
            except APIError as e:
                raise LLMError(f"stream error: {e}") from e
        finally:
            llm_active_requests.labels(endpoint=endpoint).dec()
            llm_request_duration.labels(
                model=model, endpoint=endpoint, stream="true", status=status
            ).observe(time.perf_counter() - t0)

    # ---- embeddings ----

    async def embed(
        self,
        model: str,
        inputs: list[str],
        *,
        generation_name: str = "embed",
    ) -> dict[str, Any]:
        """Devuelve {'embeddings': [[..], ..], 'prompt_eval_count': int}."""
        endpoint = "embeddings"
        status = "error"
        t0 = time.perf_counter()
        # Adjunta metadata Langfuse igual que los métodos de chat: si hay trace
        # activo (p.ej. el embed del rag_search dentro de un agent.run), la
        # generation del embedding cuelga de ese trace en vez de quedar huérfana.
        # Sin trace (indexado/watcher), `extra` queda vacío y LiteLLM registra
        # su propia traza, que es lo razonable para trabajo de fondo.
        extra: dict[str, Any] = {}
        self._attach_langfuse(extra, generation_name)
        llm_active_requests.labels(endpoint=endpoint).inc()
        try:
            try:
                resp = await self._client.embeddings.create(
                    model=model, input=inputs, **extra
                )
            except APIStatusError as e:
                raise LLMError(f"upstream {e.status_code}: {e.message}") from e
            except APIError as e:
                raise LLMError(f"network error: {e}") from e
            vectors = [list(item.embedding) for item in resp.data]
            usage = getattr(resp, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
            if prompt_tokens:
                llm_tokens_total.labels(model=model, direction="input").inc(prompt_tokens)
            status = "ok"
            return {"embeddings": vectors, "prompt_eval_count": prompt_tokens}
        finally:
            llm_active_requests.labels(endpoint=endpoint).dec()
            llm_request_duration.labels(
                model=model, endpoint=endpoint, stream="false", status=status
            ).observe(time.perf_counter() - t0)

    # ---- helpers ----

    @staticmethod
    def _attach_langfuse(payload: dict[str, Any], generation_name: str) -> None:
        """Mete metadata Langfuse en extra_body si hay trace activo.

        LiteLLM acepta `metadata` con `trace_id`, `parent_observation_id` y
        `generation_name`, y crea la generation bajo el trace correcto.
        """
        meta = get_tracer().litellm_metadata(generation_name)
        if not meta:
            return
        extra = payload.setdefault("extra_body", {})
        merged = dict(extra.get("metadata") or {})
        merged.update(meta)
        extra["metadata"] = merged

    @staticmethod
    def _build_chat_payload(req: ChatCompletionRequest, stream: bool) -> dict[str, Any]:
        msgs = [_message_to_openai(m) for m in req.messages]
        payload: dict[str, Any] = {
            "model": req.model,
            "messages": msgs,
            "stream": stream,
        }
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.top_p is not None:
            payload["top_p"] = req.top_p
        if req.max_tokens is not None:
            payload["max_tokens"] = req.max_tokens
        if req.seed is not None:
            payload["seed"] = req.seed
        if req.stop is not None:
            payload["stop"] = req.stop
        if req.presence_penalty is not None:
            payload["presence_penalty"] = req.presence_penalty
        if req.frequency_penalty is not None:
            payload["frequency_penalty"] = req.frequency_penalty
        if stream:
            # Pedimos usage stats también en el stream final si el proveedor lo soporta.
            payload["stream_options"] = {"include_usage": True}
        return payload

    @staticmethod
    def _build_tools_payload(
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any] | None,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "stream": stream,
        }
        opts = options or {}
        if "temperature" in opts:
            payload["temperature"] = opts["temperature"]
        if "top_p" in opts:
            payload["top_p"] = opts["top_p"]
        if "max_tokens" in opts or "num_predict" in opts:
            payload["max_tokens"] = opts.get("max_tokens") or opts.get("num_predict")
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload


def _message_to_openai(m: ChatMessage) -> dict[str, Any]:
    out: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.name:
        out["name"] = m.name
    return out


def _chunk_has_content(chunk: dict[str, Any]) -> bool:
    """¿Este chunk ya trae contenido útil? Sirve para fijar TTFT."""
    choices = chunk.get("choices") or []
    if not choices:
        return False
    delta = choices[0].get("delta") or {}
    if delta.get("content"):
        return True
    if delta.get("tool_calls"):
        return True
    return False


def _record_tokens(model: str, usage: dict[str, Any] | None) -> None:
    if not usage:
        return
    pt = int(usage.get("prompt_tokens") or 0)
    ct = int(usage.get("completion_tokens") or 0)
    if pt:
        llm_tokens_total.labels(model=model, direction="input").inc(pt)
    if ct:
        llm_tokens_total.labels(model=model, direction="output").inc(ct)
