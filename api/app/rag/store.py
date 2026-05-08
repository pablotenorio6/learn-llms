"""Wrapper async sobre qdrant-client.

La colección guarda un punto por chunk. Metadata (payload):
  - doc_id    : str (hash del contenido del documento)
  - source    : str (path o filename original)
  - chunk_idx : int (orden dentro del documento)
  - chunk_hash: str (hash del chunk, para idempotencia)
  - text      : str (el contenido del chunk, para devolverlo en search)
  - bytes     : int (tamaño del documento original)
  - mtime     : float (modificación, para watcher)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient, models

log = logging.getLogger(__name__)


@dataclass
class SearchHit:
    text: str
    score: float
    source: str
    doc_id: str
    chunk_idx: int


class QdrantStore:
    def __init__(self, url: str, collection: str, embed_dim: int):
        self.url = url
        self.collection = collection
        self.embed_dim = embed_dim
        self._client = AsyncQdrantClient(url=url)

    async def aclose(self) -> None:
        await self._client.close()

    async def ensure_collection(self) -> None:
        """Crea la colección si no existe, con índices secundarios para filtrar."""
        existing = await self._client.get_collections()
        names = {c.name for c in existing.collections}
        if self.collection not in names:
            await self._client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self.embed_dim,
                    distance=models.Distance.COSINE,
                ),
            )
            log.info("rag.collection_created", extra={"name": self.collection})
        # Índices para acelerar filtros / borrados por doc_id y source
        for field, schema in (
            ("doc_id", models.PayloadSchemaType.KEYWORD),
            ("source", models.PayloadSchemaType.KEYWORD),
            ("chunk_hash", models.PayloadSchemaType.KEYWORD),
        ):
            try:
                await self._client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception:
                # ya existía
                pass

    async def upsert_chunks(
        self,
        doc_id: str,
        source: str,
        chunk_texts: list[str],
        chunk_vectors: list[list[float]],
        chunk_hashes: list[str],
        extra_payload: dict[str, Any] | None = None,
    ) -> int:
        """Inserta o actualiza chunks. Devuelve cuántos se subieron."""
        if len(chunk_texts) != len(chunk_vectors) != len(chunk_hashes):
            raise ValueError("listas con longitudes distintas")
        points = []
        for i, (txt, vec, h) in enumerate(zip(chunk_texts, chunk_vectors, chunk_hashes)):
            payload = {
                "doc_id": doc_id,
                "source": source,
                "chunk_idx": i,
                "chunk_hash": h,
                "text": txt,
                **(extra_payload or {}),
            }
            # ID determinista por chunk_hash → re-indexar el mismo doc no duplica
            point_id = str(uuid.uuid5(uuid.NAMESPACE_OID, h))
            points.append(
                models.PointStruct(id=point_id, vector=vec, payload=payload)
            )
        await self._client.upsert(collection_name=self.collection, points=points)
        return len(points)

    async def delete_doc(self, doc_id: str) -> int:
        """Borra todos los chunks de un documento. Devuelve nº borrados (aprox)."""
        # Contar antes (Qdrant no devuelve count en delete por filter)
        n = await self.count_by(doc_id=doc_id)
        await self._client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
                )
            ),
        )
        return n

    async def delete_by_source(self, source: str) -> int:
        n = await self.count_by(source=source)
        await self._client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))]
                )
            ),
        )
        return n

    async def count_by(self, doc_id: str | None = None, source: str | None = None) -> int:
        must = []
        if doc_id:
            must.append(models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)))
        if source:
            must.append(models.FieldCondition(key="source", match=models.MatchValue(value=source)))
        if not must:
            res = await self._client.count(collection_name=self.collection, exact=True)
            return res.count
        res = await self._client.count(
            collection_name=self.collection,
            count_filter=models.Filter(must=must),
            exact=True,
        )
        return res.count

    async def list_documents(self) -> list[dict[str, Any]]:
        """Devuelve resumen agrupando por doc_id (source, n_chunks, doc_id)."""
        # Qdrant no soporta GROUP BY directo; hacemos scroll y agregamos en memoria.
        # Para colecciones moderadas es suficiente.
        offset = None
        groups: dict[str, dict[str, Any]] = {}
        while True:
            points, offset = await self._client.scroll(
                collection_name=self.collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                pl = p.payload or {}
                did = pl.get("doc_id", "?")
                g = groups.setdefault(did, {"doc_id": did, "source": pl.get("source", "?"), "chunks": 0})
                g["chunks"] += 1
            if offset is None:
                break
        return sorted(groups.values(), key=lambda g: g["source"])

    async def search(self, vector: list[float], top_k: int = 5) -> list[SearchHit]:
        res = await self._client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        hits: list[SearchHit] = []
        for p in res.points:
            pl = p.payload or {}
            hits.append(SearchHit(
                text=pl.get("text", ""),
                score=p.score,
                source=pl.get("source", "?"),
                doc_id=pl.get("doc_id", "?"),
                chunk_idx=int(pl.get("chunk_idx", 0)),
            ))
        return hits
