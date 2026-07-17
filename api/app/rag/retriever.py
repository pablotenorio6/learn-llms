"""Recuperador: query string → (denso + BM25) → fusión RRF → SearchHits.

Híbrido: la rama densa (embeddings nomic en Qdrant) capta similitud semántica;
la rama léxica (BM25 en memoria) capta coincidencia exacta de términos que el
embedding difumina (nombres, siglas, vocabulario "meta" ausente del texto).
Se fusionan con Reciprocal Rank Fusion ponderada, que combina por POSICIÓN en
cada ranking —no por magnitud de score— y por eso es robusta aunque las escalas
de coseno y BM25 no sean comparables.

El índice BM25 vive en memoria y se reconstruye cuando `store.mutations` cambia
(alta/baja de chunks). Para el tamaño de KB de este proyecto es de sobra.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from app.config import get_settings
from app.observability import get_tracer, rag_chunks_returned, rag_retrieval_duration
from app.rag.bm25 import BM25Index
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
        self._bm25 = BM25Index()
        self._bm25_version = -1
        self._corpus: dict[str, SearchHit] = {}

    async def _ensure_bm25(self) -> None:
        """Reconstruye el índice BM25 si la colección cambió desde la última vez."""
        if self.store.mutations == self._bm25_version:
            return
        chunks = await self.store.scroll_all()
        self._corpus = {c.id: c for c in chunks}
        self._bm25.fit([c.id for c in chunks], [c.text for c in chunks])
        self._bm25_version = self.store.mutations
        log.info("rag.bm25_rebuilt", extra={"chunks": len(chunks), "version": self._bm25_version})

    def _fuse(
        self,
        dense: list[SearchHit],
        bm25: list[tuple[str, float]],
        top_k: int,
        rrf_k: int,
        w_dense: float,
        w_bm25: float,
    ) -> list[SearchHit]:
        """Reciprocal Rank Fusion ponderada sobre los ids de ambos rankings."""
        scores: dict[str, float] = {}
        for rank, h in enumerate(dense):
            scores[h.id] = scores.get(h.id, 0.0) + w_dense / (rrf_k + rank)
        for rank, (cid, _) in enumerate(bm25):
            scores[cid] = scores.get(cid, 0.0) + w_bm25 / (rrf_k + rank)

        # Payload de cada id: preferimos el SearchHit denso (trae texto), si no
        # tiramos del corpus cacheado (cubre los rescatados solo por BM25).
        dense_by_id = {h.id: h for h in dense}
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        out: list[SearchHit] = []
        for cid, fused in ordered:
            base = dense_by_id.get(cid) or self._corpus.get(cid)
            if base is None:
                continue
            out.append(SearchHit(
                text=base.text, score=round(fused, 6), source=base.source,
                doc_id=base.doc_id, chunk_idx=base.chunk_idx, id=cid,
            ))
        return out

    async def query(self, text: str, top_k: int = 5) -> RetrievalResult:
        settings = get_settings()
        tracer = get_tracer()
        t0 = time.perf_counter()
        with tracer.span(
            "rag.retrieval",
            input=text if tracer.log_payloads else None,
            metadata={"top_k": top_k, "query_len": len(text), "hybrid": settings.rag_hybrid_enabled},
        ) as span:
            if not settings.rag_hybrid_enabled:
                vec = await self.embedder.embed_one(text, task="query")
                hits = await self.store.search(vector=vec, top_k=top_k)
            else:
                cand_k = max(top_k, settings.rag_candidate_k)

                # Las dos ramas no dependen entre sí: la densa necesita el
                # embedding de la query (roundtrip a LiteLLM→Ollama) y la
                # léxica el índice BM25 (que puede tocar reconstruirse con un
                # scroll a Qdrant). Solaparlas ahorra el mínimo de las dos.
                async def _dense_branch() -> list[SearchHit]:
                    vec = await self.embedder.embed_one(text, task="query")
                    return await self.store.search(vector=vec, top_k=cand_k)

                async def _lexical_branch() -> list[tuple[str, float]]:
                    await self._ensure_bm25()
                    return self._bm25.search(text, cand_k)

                dense_hits, bm25_hits = await asyncio.gather(
                    _dense_branch(), _lexical_branch()
                )
                hits = self._fuse(
                    dense_hits, bm25_hits, top_k,
                    settings.rag_rrf_k, settings.rag_dense_weight, settings.rag_bm25_weight,
                )

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
