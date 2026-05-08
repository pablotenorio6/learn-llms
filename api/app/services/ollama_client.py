"""Cliente async para la API de Ollama.

Documentación: https://github.com/ollama/ollama/blob/main/docs/api.md

Traducimos entre el dialecto OpenAI y el de Ollama. Lo dejamos explícito
en helpers para que sea fácil ver qué campos viajan.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

from app.models import ChatCompletionRequest, ChatMessage

log = logging.getLogger(__name__)


class OllamaError(Exception):
    """Cualquier fallo del lado de Ollama (HTTP, parseo, etc.)."""


class OllamaClient:
    def __init__(self, base_url: str, timeout: int = 600, keep_alive: str = "10m"):
        self.base_url = base_url.rstrip("/")
        self.keep_alive = keep_alive
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ---- low level ----

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            r = await self._client.post(path, json=payload)
        except httpx.HTTPError as e:
            raise OllamaError(f"network error: {e}") from e
        if r.status_code >= 400:
            raise OllamaError(f"ollama returned {r.status_code}: {r.text[:500]}")
        return r.json()

    async def _post_stream(
        self, path: str, payload: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        try:
            async with self._client.stream("POST", path, json=payload) as r:
                if r.status_code >= 400:
                    body = await r.aread()
                    raise OllamaError(
                        f"ollama returned {r.status_code}: {body[:500].decode(errors='replace')}"
                    )
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        log.warning("ollama.bad_jsonl", extra={"line": line[:200]})
                        continue
        except httpx.HTTPError as e:
            raise OllamaError(f"stream error: {e}") from e

    # ---- high level ----

    async def list_models(self) -> list[dict[str, Any]]:
        try:
            r = await self._client.get("/api/tags")
        except httpx.HTTPError as e:
            raise OllamaError(f"network error: {e}") from e
        if r.status_code >= 400:
            raise OllamaError(f"ollama returned {r.status_code}: {r.text[:500]}")
        return r.json().get("models", [])

    async def chat(self, req: ChatCompletionRequest) -> dict[str, Any]:
        """Llamada no-streaming. Devuelve el dict JSON crudo de Ollama."""
        payload = self._build_chat_payload(req, stream=False)
        return await self._post("/api/chat", payload)

    async def chat_stream(self, req: ChatCompletionRequest) -> AsyncIterator[dict[str, Any]]:
        payload = self._build_chat_payload(req, stream=True)
        async for chunk in self._post_stream("/api/chat", payload):
            yield chunk

    async def embed(self, model: str, inputs: list[str]) -> dict[str, Any]:
        payload = {"model": model, "input": inputs, "keep_alive": self.keep_alive}
        return await self._post("/api/embed", payload)

    # ---- traducción OpenAI → Ollama ----

    def _build_chat_payload(self, req: ChatCompletionRequest, stream: bool) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if req.temperature is not None:
            options["temperature"] = req.temperature
        if req.top_p is not None:
            options["top_p"] = req.top_p
        if req.max_tokens is not None:
            options["num_predict"] = req.max_tokens
        if req.seed is not None:
            options["seed"] = req.seed
        if req.stop is not None:
            options["stop"] = [req.stop] if isinstance(req.stop, str) else req.stop
        if req.presence_penalty is not None:
            options["presence_penalty"] = req.presence_penalty
        if req.frequency_penalty is not None:
            options["frequency_penalty"] = req.frequency_penalty

        return {
            "model": req.model,
            "messages": [_message_to_ollama(m) for m in req.messages],
            "stream": stream,
            "keep_alive": self.keep_alive,
            "options": options,
        }


def _message_to_ollama(m: ChatMessage) -> dict[str, Any]:
    out: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.name:
        out["name"] = m.name
    return out


# ---- Tool calling (Fase 4) ----

async def _chat_with_tools_impl(client, model, messages, tools, options=None):
    """Llamada no-streaming con tools. Devuelve el dict crudo de Ollama."""
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": False,
        "keep_alive": client.keep_alive,
        "options": options or {},
    }
    return await client._post("/api/chat", payload)


# Añadimos como método al vuelo (más simple que volver a definir la clase)
OllamaClient.chat_with_tools = lambda self, model, messages, tools, options=None: _chat_with_tools_impl(
    self, model, messages, tools, options
)


async def _chat_with_tools_stream_impl(client, model, messages, tools, options=None):
    """Streaming con tools. Yields chunks tal cual los devuelve Ollama (JSONL).
    
    Cada chunk puede traer message.content (token), y en done=true puede traer
    message.tool_calls. El consumidor acumula content y revisa tool_calls al cerrar.
    """
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": True,
        "keep_alive": client.keep_alive,
        "options": options or {},
    }
    async for chunk in client._post_stream("/api/chat", payload):
        yield chunk


async def _chat_with_tools_stream_method(self, model, messages, tools, options=None):
    async for c in _chat_with_tools_stream_impl(self, model, messages, tools, options):
        yield c

OllamaClient.chat_with_tools_stream = _chat_with_tools_stream_method
