"""Recuperador: query string → embedding → search → SearchHits formateados."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.rag.embedder import Embedder
from app.rag.store import QdrantStore, SearchHit

log = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    hits: list[SearchHit]
    query: str

    def as_system_message(self) -> str:
        """Formatea los hits como un mensaje system listo para inyectar en el chat."""
        if not self.hits:
            return (
                "No se encontró información relevante en la base de conocimiento. "
                "Responde con tu conocimiento general y di que no encontraste fuentes locales."
            )
        parts = [
            "Tienes acceso a los siguientes fragmentos recuperados de la base de conocimiento "
            "del usuario. Úsalos para responder. Si citas un dato, indica la fuente entre corchetes "
            "como [source]. Si los fragmentos no responden a la pregunta, dilo explícitamente.\n",
        ]
        for i, h in enumerate(self.hits, 1):
            parts.append(f"--- [{i}] source: {h.source} (score={h.score:.3f}) ---")
            parts.append(h.text)
            parts.append("")
        return "\n".join(parts)


class Retriever:
    def __init__(self, store: QdrantStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    async def query(self, text: str, top_k: int = 5) -> RetrievalResult:
        vec = await self.embedder.embed_one(text)
        hits = await self.store.search(vector=vec, top_k=top_k)
        log.info("rag.retrieved", extra={"query_len": len(text), "hits": len(hits)})
        return RetrievalResult(hits=hits, query=text)
