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
    def __init__(self, client: LLMClient, model: str, batch_size: int = 32):
        self.client = client
        self.model = model
        self.batch_size = batch_size

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
        vs = await self.embed([text], task=task)
        return vs[0]


def _batched(items: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(items), n):
        yield items[i:i + n]
