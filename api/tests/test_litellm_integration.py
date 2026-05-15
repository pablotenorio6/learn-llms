"""Tests E2E del refactor a LiteLLM.

Mockea el LLMClient inyectado en app.state.llm para cubrir:
  - chat no-streaming
  - chat streaming
  - agent loop trivial (sin tools)
  - agent loop con 1 tool
  - agent loop con tools paralelas

Se ejecuta con:
    python -m tests.test_litellm_integration
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, AsyncIterator

# Setup env antes de importar app.
os.environ.setdefault("LITELLM_BASE_URL", "http://x:1")
os.environ.setdefault("LITELLM_MASTER_KEY", "sk-test")
os.environ.setdefault("QDRANT_HOST", "http://x:6333")
os.environ.setdefault("RAG_WATCHER_ENABLED", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


class FakeLLM:
    """Doble de LLMClient con respuestas predefinidas por método."""

    def __init__(self):
        self.chat_response: dict[str, Any] | None = None
        self.chat_stream_chunks: list[dict[str, Any]] = []
        self.tools_stream_iterations: list[list[dict[str, Any]]] = []
        self._iter_idx = 0
        self.embed_response: dict[str, Any] | None = None
        self.models_response: list[dict[str, Any]] = []

    async def aclose(self):
        return None

    async def list_models(self):
        return list(self.models_response)

    async def chat(self, req):
        assert self.chat_response is not None
        return self.chat_response

    async def chat_stream(self, req) -> AsyncIterator[dict[str, Any]]:
        for c in self.chat_stream_chunks:
            yield c

    async def chat_with_tools(self, model, messages, tools, options=None):
        assert self.tools_stream_iterations
        chunks = self.tools_stream_iterations[self._iter_idx]
        self._iter_idx += 1
        # Simulamos agregación devolviendo solo el último chunk como respuesta.
        return chunks[-1]

    async def chat_with_tools_stream(self, model, messages, tools, options=None) -> AsyncIterator[dict[str, Any]]:
        chunks = self.tools_stream_iterations[self._iter_idx]
        self._iter_idx += 1
        for c in chunks:
            yield c

    async def embed(self, model, inputs):
        return self.embed_response or {"embeddings": [[0.0] * 3 for _ in inputs], "prompt_eval_count": len(inputs)}


# ---- helpers para construir chunks OpenAI ----

def chunk_content(piece: str, finish: str | None = None) -> dict[str, Any]:
    delta: dict[str, Any] = {"content": piece}
    return {
        "id": "c1",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def chunk_tool_call(index: int, id: str | None, name: str | None, args_piece: str, finish: str | None = None) -> dict[str, Any]:
    fn: dict[str, Any] = {}
    if name is not None:
        fn["name"] = name
    fn["arguments"] = args_piece
    tc: dict[str, Any] = {"index": index, "function": fn}
    if id is not None:
        tc["id"] = id
        tc["type"] = "function"
    return {
        "id": "c1",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"tool_calls": [tc]}, "finish_reason": finish}],
    }


def chunk_final(finish: str = "stop") -> dict[str, Any]:
    return {
        "id": "c1",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
    }


# ---- tests ----

PASS = 0
FAIL = 0


def check(label: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}  {detail}")


def install_fake_llm(client: TestClient) -> FakeLLM:
    fake = FakeLLM()
    client.app.state.llm = fake
    return fake


def test_chat_nonstream():
    print("\n[test] chat no-streaming")
    with TestClient(app) as client:
        fake = install_fake_llm(client)
        fake.chat_response = {
            "id": "chatcmpl-x",
            "model": "qwen-local",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hola mundo"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
        r = client.post("/v1/chat/completions", json={
            "model": "qwen-local",
            "messages": [{"role": "user", "content": "hola"}],
        })
        check("status 200", r.status_code == 200, str(r.status_code))
        data = r.json()
        check("content match", data["choices"][0]["message"]["content"] == "hola mundo")
        check("usage propagada", data["usage"]["total_tokens"] == 7)


def test_chat_stream():
    print("\n[test] chat streaming")
    with TestClient(app) as client:
        fake = install_fake_llm(client)
        fake.chat_stream_chunks = [
            chunk_content("Hola "),
            chunk_content("mundo"),
            chunk_final("stop"),
        ]
        with client.stream("POST", "/v1/chat/completions", json={
            "model": "qwen-local",
            "messages": [{"role": "user", "content": "saluda"}],
            "stream": True,
        }) as r:
            check("status 200", r.status_code == 200, str(r.status_code))
            collected = []
            for line in r.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                if payload == "[DONE]":
                    collected.append("__DONE__")
                    continue
                collected.append(json.loads(payload))
        # Esperamos: role-chunk, "Hola ", "mundo", finish, DONE
        texts = [c["choices"][0]["delta"].get("content") for c in collected if isinstance(c, dict)]
        check("contiene 'Hola '", "Hola " in texts)
        check("contiene 'mundo'", "mundo" in texts)
        check("[DONE] al final", collected[-1] == "__DONE__")


def test_agent_trivial_no_tools():
    print("\n[test] agent trivial (sin tools)")
    with TestClient(app) as client:
        fake = install_fake_llm(client)
        fake.tools_stream_iterations = [[
            chunk_content("una manzana es roja", finish="stop"),
        ]]
        with client.stream("POST", "/v1/agents/run", json={
            "model": "qwen-local",
            "messages": [{"role": "user", "content": "¿qué color tiene una manzana?"}],
        }) as r:
            check("status 200", r.status_code == 200, str(r.status_code))
            events = _parse_sse(r)
        types = [e.get("type") for e in events if isinstance(e, dict)]
        check("hay iteration", "iteration" in types)
        check("hay final", "final" in types)
        check("sin tool_call", "tool_call" not in types)
        final = [e for e in events if isinstance(e, dict) and e.get("type") == "final"][0]
        check("final content correcto", "manzana" in final["content"])


def test_agent_one_tool():
    print("\n[test] agent con 1 tool (rag_search)")
    with TestClient(app) as client:
        fake = install_fake_llm(client)
        # Iteración 1: el modelo emite un tool_call por streaming.
        # OpenAI parte el JSON de arguments en deltas. Probamos esa acumulación.
        fake.tools_stream_iterations = [
            [
                chunk_tool_call(0, "call_abc", "rag_search", '{"query"'),
                chunk_tool_call(0, None, None, ': "memorias"'),
                chunk_tool_call(0, None, None, ', "top_k": 3}', finish="tool_calls"),
            ],
            [
                chunk_content("Según tus notas, ", finish=None),
                chunk_content("encontré algo.", finish="stop"),
            ],
        ]
        # ctx.rag debe existir para que la tool no falle: dejamos un retriever fake.
        class FakeRetriever:
            async def query(self, q, top_k=5):
                class R:
                    query = q
                    hits = []
                    def as_system_message(self_):
                        return ""
                return R()
        client.app.state.rag = {"retriever": FakeRetriever(), "store": None, "indexer": None}
        with client.stream("POST", "/v1/agents/run", json={
            "model": "qwen-local",
            "messages": [{"role": "user", "content": "busca en mis notas"}],
            "tools_allowed": ["rag_search"],
        }) as r:
            check("status 200", r.status_code == 200, str(r.status_code))
            events = _parse_sse(r)
        types = [e.get("type") for e in events if isinstance(e, dict)]
        tool_calls = [e for e in events if isinstance(e, dict) and e.get("type") == "tool_call"]
        tool_results = [e for e in events if isinstance(e, dict) and e.get("type") == "tool_result"]
        finals = [e for e in events if isinstance(e, dict) and e.get("type") == "final"]
        check("hay 1 tool_call", len(tool_calls) == 1, str(len(tool_calls)))
        check("tool_call.name correcto", tool_calls[0]["name"] == "rag_search" if tool_calls else False)
        check("arguments parseados", tool_calls[0]["arguments"] == {"query": "memorias", "top_k": 3} if tool_calls else False, str(tool_calls[0].get("arguments") if tool_calls else None))
        check("hay 1 tool_result", len(tool_results) == 1, str(len(tool_results)))
        check("hay final", len(finals) == 1)
        if finals:
            check("final content tras tool", "encontré" in finals[0]["content"])


def test_agent_ollama_style_tool_call():
    """Ollama vía LiteLLM emite todo el tool_call en un chunk y finish_reason='stop'.
    El loop debe ejecutar la tool igualmente — no fiarse de finish_reason."""
    print("\n[test] agent con dialect Ollama (finish='stop' aunque haya tool_calls)")
    with TestClient(app) as client:
        fake = install_fake_llm(client)
        fake.tools_stream_iterations = [
            [
                # Un único chunk con tool_call completo y finish=stop (lo que hace Ollama).
                chunk_tool_call(0, "call_ollama", "rag_search", '{"query": "Napoleon"}', finish="stop"),
            ],
            [
                chunk_content("Napoleón fue…", finish="stop"),
            ],
        ]
        class FakeRetriever:
            async def query(self, q, top_k=5):
                class R:
                    query = q
                    hits = []
                    def as_system_message(self_):
                        return ""
                return R()
        client.app.state.rag = {"retriever": FakeRetriever(), "store": None, "indexer": None}
        with client.stream("POST", "/v1/agents/run", json={
            "model": "llama-local",
            "messages": [{"role": "user", "content": "quién fue Napoleón"}],
            "tools_allowed": ["rag_search"],
        }) as r:
            check("status 200", r.status_code == 200, str(r.status_code))
            events = _parse_sse(r)
        tool_calls = [e for e in events if isinstance(e, dict) and e.get("type") == "tool_call"]
        finals = [e for e in events if isinstance(e, dict) and e.get("type") == "final"]
        check("tool_call ejecutado a pesar de finish=stop", len(tool_calls) == 1, str(len(tool_calls)))
        check("hay final tras la tool", len(finals) == 1 and "Napoleón" in finals[0]["content"] if finals else False)


def test_agent_parallel_tools():
    print("\n[test] agent con tools paralelas (2 a la vez)")
    with TestClient(app) as client:
        fake = install_fake_llm(client)
        fake.tools_stream_iterations = [
            [
                # Tool 0 y tool 1 entrelazadas.
                chunk_tool_call(0, "call_a", "rag_search", '{"query": "A"}'),
                chunk_tool_call(1, "call_b", "rag_search", '{"query"'),
                chunk_tool_call(1, None, None, ': "B"}', finish="tool_calls"),
            ],
            [
                chunk_content("Respuesta final.", finish="stop"),
            ],
        ]
        class FakeRetriever:
            async def query(self, q, top_k=5):
                class R:
                    query = q
                    hits = []
                    def as_system_message(self_):
                        return ""
                return R()
        client.app.state.rag = {"retriever": FakeRetriever(), "store": None, "indexer": None}
        with client.stream("POST", "/v1/agents/run", json={
            "model": "qwen-local",
            "messages": [{"role": "user", "content": "compara A y B en mis notas"}],
            "tools_allowed": ["rag_search"],
        }) as r:
            check("status 200", r.status_code == 200, str(r.status_code))
            events = _parse_sse(r)
        tool_calls = [e for e in events if isinstance(e, dict) and e.get("type") == "tool_call"]
        tool_results = [e for e in events if isinstance(e, dict) and e.get("type") == "tool_result"]
        check("hay 2 tool_calls", len(tool_calls) == 2, str(len(tool_calls)))
        check("hay 2 tool_results", len(tool_results) == 2, str(len(tool_results)))
        if len(tool_calls) == 2:
            queries = sorted([tc["arguments"]["query"] for tc in tool_calls])
            check("queries correctos", queries == ["A", "B"], str(queries))


def test_embeddings_endpoint():
    print("\n[test] /v1/embeddings via LLMClient.embed")
    with TestClient(app) as client:
        fake = install_fake_llm(client)
        fake.embed_response = {"embeddings": [[0.1, 0.2], [0.3, 0.4]], "prompt_eval_count": 7}
        r = client.post("/v1/embeddings", json={
            "model": "nomic-embed",
            "input": ["foo", "bar"],
        })
        check("status 200", r.status_code == 200, str(r.status_code))
        data = r.json()
        check("2 items", len(data["data"]) == 2)
        check("vector ok", data["data"][0]["embedding"] == [0.1, 0.2])


def test_models_endpoint():
    print("\n[test] /v1/models via LiteLLM proxy")
    with TestClient(app) as client:
        fake = install_fake_llm(client)
        fake.models_response = [
            {"id": "qwen-local", "object": "model", "created": 0, "owned_by": "openai"},
            {"id": "gpt-4o-mini", "object": "model", "created": 0, "owned_by": "openai"},
        ]
        r = client.get("/v1/models")
        check("status 200", r.status_code == 200, str(r.status_code))
        data = r.json()
        ids = sorted([m["id"] for m in data["data"]])
        check("aliases presentes", ids == ["gpt-4o-mini", "qwen-local"], str(ids))


def _parse_sse(response) -> list[Any]:
    out = []
    for line in response.iter_lines():
        if not line or not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            out.append("__DONE__")
            continue
        try:
            out.append(json.loads(payload))
        except json.JSONDecodeError:
            out.append({"raw": payload})
    return out


if __name__ == "__main__":
    test_chat_nonstream()
    test_chat_stream()
    test_agent_trivial_no_tools()
    test_agent_one_tool()
    test_agent_ollama_style_tool_call()
    test_agent_parallel_tools()
    test_embeddings_endpoint()
    test_models_endpoint()
    print(f"\n=== {PASS} passed, {FAIL} failed ===")
    sys.exit(0 if FAIL == 0 else 1)
