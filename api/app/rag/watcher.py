"""Watcher de carpeta de docs.

Watchdog corre en su propio thread (no es async). Encolamos eventos en
un asyncio.Queue desde el callback síncrono usando run_coroutine_threadsafe,
y un consumidor async drena la cola debouncedo (varios eventos seguidos
sobre el mismo path se colapsan).

Hace una pasada inicial al arrancar para indexar lo que ya existe.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from app.rag.indexer import Indexer
from app.rag.parsers import SUPPORTED_EXTENSIONS

log = logging.getLogger(__name__)


class _Handler(FileSystemEventHandler):
    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        self.queue = queue
        self.loop = loop

    def _push(self, kind: str, src_path: str):
        path = Path(src_path)
        if path.is_dir():
            return
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return
        # Encolar desde un thread externo
        asyncio.run_coroutine_threadsafe(self.queue.put((kind, str(path))), self.loop)

    def on_created(self, event: FileSystemEvent):
        if not event.is_directory:
            self._push("upsert", event.src_path)

    def on_modified(self, event: FileSystemEvent):
        if not event.is_directory:
            self._push("upsert", event.src_path)

    def on_deleted(self, event: FileSystemEvent):
        if not event.is_directory:
            self._push("delete", event.src_path)

    def on_moved(self, event: FileSystemEvent):
        if not event.is_directory:
            self._push("delete", event.src_path)
            self._push("upsert", event.dest_path)


class DocsWatcher:
    DEBOUNCE_MS = 500

    def __init__(self, docs_dir: Path, indexer: Indexer):
        self.docs_dir = docs_dir
        self.indexer = indexer
        self._observer: Observer | None = None
        self._queue: asyncio.Queue | None = None
        self._consumer: asyncio.Task | None = None

    async def start(self) -> None:
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self._observer = Observer()
        self._observer.schedule(
            _Handler(self._queue, loop),
            str(self.docs_dir),
            recursive=True,
        )
        self._observer.start()
        self._consumer = asyncio.create_task(self._consume(), name="rag-watcher-consumer")

        # Pasada inicial: indexa lo que ya hay
        asyncio.create_task(self._initial_scan(), name="rag-watcher-initial")

    async def stop(self) -> None:
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
        if self._consumer:
            self._consumer.cancel()
            try:
                await self._consumer
            except asyncio.CancelledError:
                pass

    async def _initial_scan(self) -> None:
        n = 0
        for path in self.docs_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            try:
                await self.indexer.index_path(path)
                n += 1
            except Exception as e:
                log.warning("rag.initial_index_failed", extra={"path": str(path), "err": str(e)})
        log.info("rag.initial_scan_done", extra={"indexed": n})

    async def _consume(self) -> None:
        """Drena la cola con debounce: si el mismo path llega varias veces seguidas, colapsa."""
        assert self._queue is not None
        pending: dict[str, str] = {}  # path -> last kind
        while True:
            try:
                kind, path = await self._queue.get()
                pending[path] = kind
                # Acumular más eventos durante DEBOUNCE_MS
                while True:
                    try:
                        kind2, path2 = await asyncio.wait_for(
                            self._queue.get(), timeout=self.DEBOUNCE_MS / 1000.0
                        )
                        pending[path2] = kind2
                    except asyncio.TimeoutError:
                        break

                # Procesar
                for p, k in list(pending.items()):
                    try:
                        if k == "delete":
                            n = await self.indexer.remove_by_source(p)
                            log.info("rag.watcher_delete", extra={"path": p, "removed_chunks": n})
                        elif k == "upsert":
                            path_obj = Path(p)
                            if not path_obj.exists():
                                # delete encadenado tras modify
                                await self.indexer.remove_by_source(p)
                            else:
                                await self.indexer.index_path(path_obj)
                                log.info("rag.watcher_upsert", extra={"path": p})
                    except Exception as e:
                        log.warning("rag.watcher_failed", extra={"path": p, "err": str(e)})
                pending.clear()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.exception("rag.watcher_loop_error", extra={"err": str(e)})
