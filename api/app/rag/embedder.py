"""Wrapper sobre LLMClient.embed con batching simple.

Prefijos de tarea (nomic-embed-text): el modelo se entrenó con una instrucción
incrustada en el texto y SIN ella la recuperación se degrada mucho (scores
apelotonados, vecinos semánticamente erróneos). El protocolo es:
  - documentos al indexar → "search_document: <texto>"
  - queries al buscar      → "search_query: <texto>"
Ollama no los inyecta: los ponemos aquí. Query y documento DEBEN usar el prefijo
que les corresponde; mezclar (o no poner ninguno) rompe la comparación. Otros
embedders (openai, bge…) no usan este esquema, así que solo se aplica a nomic.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Iterable

from app.services.llm_client import LLMClient

log = logging.getLogger(__name__)

_NOMIC_PREFIX = {
    "query": "search_query: ",
    "document": "search_document: ",
}


def _needs_nomic_prefix(model: str) -> bool:
    return "nomic" in model.lower()


class Embedder:
    def __init__(
        self,
        client: LLMClient,
        model: str,
        batch_size: int = 32,
        query_cache_size: int = 128,
    ):
        self.client = client
        self.model = model
        self.batch_size = batch_size
        # LRU de embeddings de QUERY: la misma pregunta repetida (usuario que
        # reformula, agente que reintenta, evals) no vuelve a pagar el roundtrip
        # LiteLLM→Ollama. Solo queries: el embedding de una query no depende del
        # corpus, así que no hay que invalidar nada al indexar. Los documentos
        # no se cachean (se embeben una vez al indexar y ya son idempotentes).
        self._query_cache_size = max(0, query_cache_size)
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()

    def _prepare(self, text: str, task: str) -> str:
        if _needs_nomic_prefix(self.model):
            return _NOMIC_PREFIX.get(task, _NOMIC_PREFIX["document"]) + text
        return text

    async def embed(self, texts: list[str], task: str = "document") -> list[list[float]]:
        """Devuelve un embedding por cada texto, en orden.

        task: "document" (al indexar) o "query" (al buscar). Determina el
        prefijo de tarea para modelos nomic; ignorado para el resto.
        """
        if not texts:
            return []
        prepared = [self._prepare(t, task) for t in texts]
        out: list[list[float]] = []
        for batch in _batched(prepared, self.batch_size):
            res = await self.client.embed(self.model, list(batch))
            vectors = res.get("embeddings") or []
            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"embedder: pedí {len(batch)} embeddings, recibí {len(vectors)}"
                )
            out.extend(vectors)
        return out

    async def embed_one(self, text: str, task: str = "query") -> list[float]:
        use_cache = task == "query" and self._query_cache_size > 0
        key = self._prepare(text, task)
        if use_cache:
            cached = self._query_cache.get(key)
            if cached is not None:
                self._query_cache.move_to_end(key)
                return cached
        vs = await self.embed([text], task=task)
        if use_cache:
            self._query_cache[key] = vs[0]
            if len(self._query_cache) > self._query_cache_size:
                self._query_cache.popitem(last=False)
        return vs[0]


def _batched(items: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(items), n):
        yield items[i:i + n]
