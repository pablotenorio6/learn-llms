"""Cliente async contra LiteLLM proxy.

La API ya no habla directamente con Ollama / OpenAI / Anthropic. Habla con
LiteLLM en formato OpenAI puro, y LiteLLM enruta al backend correcto según
el alias del modelo declarado en litellm-config.yaml.

Usamos el SDK oficial de OpenAI (AsyncOpenAI) apuntado a la base_url del
proxy. Esto nos da streaming SSE bien parseado y typed-objects en vez de
dicts a pelo.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from openai import APIError, APIStatusError, AsyncOpenAI

from app.models import ChatCompletionRequest, ChatMessage

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

    async def chat(self, req: ChatCompletionRequest) -> dict[str, Any]:
        """Chat completions no-streaming. Devuelve el dict (model_dump) crudo."""
        payload = self._build_chat_payload(req, stream=False)
        try:
            resp = await self._client.chat.completions.create(**payload)
        except APIStatusError as e:
            raise LLMError(f"upstream {e.status_code}: {e.message}") from e
        except APIError as e:
            raise LLMError(f"network error: {e}") from e
        return resp.model_dump()

    async def chat_stream(self, req: ChatCompletionRequest) -> AsyncIterator[dict[str, Any]]:
        """Chat completions con streaming. Yields chunk.model_dump() por chunk."""
        payload = self._build_chat_payload(req, stream=True)
        try:
            stream = await self._client.chat.completions.create(**payload)
            async for chunk in stream:
                yield chunk.model_dump()
        except APIStatusError as e:
            raise LLMError(f"upstream {e.status_code}: {e.message}") from e
        except APIError as e:
            raise LLMError(f"stream error: {e}") from e

    # ---- chat con tools ----

    async def chat_with_tools(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Chat con tool calling, no-streaming."""
        payload = self._build_tools_payload(model, messages, tools, options, stream=False)
        try:
            resp = await self._client.chat.completions.create(**payload)
        except APIStatusError as e:
            raise LLMError(f"upstream {e.status_code}: {e.message}") from e
        except APIError as e:
            raise LLMError(f"network error: {e}") from e
        return resp.model_dump()

    async def chat_with_tools_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Chat con tool calling, streaming. Yields chunks en formato OpenAI."""
        payload = self._build_tools_payload(model, messages, tools, options, stream=True)
        try:
            stream = await self._client.chat.completions.create(**payload)
            async for chunk in stream:
                yield chunk.model_dump()
        except APIStatusError as e:
            raise LLMError(f"upstream {e.status_code}: {e.message}") from e
        except APIError as e:
            raise LLMError(f"stream error: {e}") from e

    # ---- embeddings ----

    async def embed(self, model: str, inputs: list[str]) -> dict[str, Any]:
        """Devuelve {'embeddings': [[..], ..], 'prompt_eval_count': int}.

        Devolvemos {embeddings, prompt_eval_count} para que Embedder y el
        endpoint /v1/embeddings consuman la misma forma con independencia
        del backend.
        """
        try:
            resp = await self._client.embeddings.create(model=model, input=inputs)
        except APIStatusError as e:
            raise LLMError(f"upstream {e.status_code}: {e.message}") from e
        except APIError as e:
            raise LLMError(f"network error: {e}") from e
        vectors = [list(item.embedding) for item in resp.data]
        usage = getattr(resp, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        return {"embeddings": vectors, "prompt_eval_count": prompt_tokens}

    # ---- helpers ----

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
