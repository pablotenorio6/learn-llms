"""Recuperador: query string → embedding → search → SearchHits formateados."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from app.observability import get_tracer, rag_chunks_returned, rag_retrieval_duration
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
        tracer = get_tracer()
        t0 = time.perf_counter()
        with tracer.span(
            "rag.retrieval",
            input=text if tracer.log_payloads else None,
            metadata={"top_k": top_k, "query_len": len(text)},
        ) as span:
            vec = await self.embedder.embed_one(text)
            hits = await self.store.search(vector=vec, top_k=top_k)
            duration = time.perf_counter() - t0
            rag_retrieval_duration.observe(duration)
            rag_chunks_returned.observe(len(hits))
            log.info("rag.retrieved", extra={"query_len": len(text), "hits": len(hits)})
            if tracer.log_payloads:
                span.update(
                    output=[{"source": h.source, "score": h.score, "text_preview": h.text[:200]} for h in hits],
                    metadata={"chunks_returned": len(hits), "duration_s": round(duration, 4)},
                )
            return RetrievalResult(hits=hits, query=text)
