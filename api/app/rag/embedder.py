"""Wrapper sobre OllamaClient.embed con batching simple."""

from __future__ import annotations

import logging
from typing import Iterable

from app.services.ollama_client import OllamaClient

log = logging.getLogger(__name__)


class Embedder:
    def __init__(self, client: OllamaClient, model: str, batch_size: int = 32):
        self.client = client
        self.model = model
        self.batch_size = batch_size

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Devuelve un embedding por cada texto, en orden."""
        if not texts:
            return []
        out: list[list[float]] = []
        for batch in _batched(texts, self.batch_size):
            res = await self.client.embed(self.model, list(batch))
            vectors = res.get("embeddings") or []
            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"embedder: pedí {len(batch)} embeddings, recibí {len(vectors)}"
                )
            out.extend(vectors)
        return out

    async def embed_one(self, text: str) -> list[float]:
        vs = await self.embed([text])
        return vs[0]


def _batched(items: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(items), n):
        yield items[i:i + n]
