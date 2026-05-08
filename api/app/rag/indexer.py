"""Orquestador: bytes/path → texto → chunks → embeddings → store.

Idempotencia: el doc_id es sha256 del contenido. Si el contenido no cambió,
re-indexar es no-op (los chunk_hash coinciden y el upsert reemplaza por mismo id).
Si el contenido cambió, primero borramos los chunks viejos del mismo `source`.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from app.rag.chunker import chunk_text
from app.rag.embedder import Embedder
from app.rag.parsers import UnsupportedFormatError, parse_bytes, parse_file
from app.rag.store import QdrantStore

log = logging.getLogger(__name__)


@dataclass
class IndexResult:
    doc_id: str
    source: str
    chunks_indexed: int
    bytes: int


class Indexer:
    def __init__(
        self,
        store: QdrantStore,
        embedder: Embedder,
        chunk_size: int,
        chunk_overlap: int,
    ):
        self.store = store
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def index_path(self, path: Path) -> IndexResult:
        data = path.read_bytes()
        return await self._index(data, source=str(path))

    async def index_bytes(self, data: bytes, source: str) -> IndexResult:
        return await self._index(data, source=source)

    async def remove_by_source(self, source: str) -> int:
        return await self.store.delete_by_source(source)

    async def _index(self, data: bytes, source: str) -> IndexResult:
        try:
            text = parse_bytes(data, source)
        except UnsupportedFormatError as e:
            raise
        if not text.strip():
            log.warning("rag.empty_text", extra={"source": source})
            # Limpiar lo que hubiera del mismo source
            await self.store.delete_by_source(source)
            return IndexResult(doc_id="", source=source, chunks_indexed=0, bytes=len(data))

        doc_id = _sha256(data)

        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        if not chunks:
            return IndexResult(doc_id=doc_id, source=source, chunks_indexed=0, bytes=len(data))

        chunk_texts = [c.text for c in chunks]
        chunk_hashes = [_sha256(c.text.encode("utf-8")) for c in chunks]

        # Antes de subir, borra cualquier chunk previo del MISMO source (cubre cambios)
        await self.store.delete_by_source(source)

        vectors = await self.embedder.embed(chunk_texts)
        n = await self.store.upsert_chunks(
            doc_id=doc_id,
            source=source,
            chunk_texts=chunk_texts,
            chunk_vectors=vectors,
            chunk_hashes=chunk_hashes,
            extra_payload={"bytes": len(data)},
        )
        log.info(
            "rag.indexed",
            extra={"source": source, "doc_id": doc_id[:12], "chunks": n, "bytes": len(data)},
        )
        return IndexResult(doc_id=doc_id, source=source, chunks_indexed=n, bytes=len(data))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
