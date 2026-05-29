"""Cliente Langfuse: trazas LLM con jerarquía trace → span → generation.

Diseño:

- Singleton inicializado en lifespan (`init_tracer`), destruido en shutdown
  (`shutdown_tracer`) para flushear el buffer del SDK.
- Si Langfuse está deshabilitado por config (o le faltan keys), devolvemos un
  `NoopTracer` con la misma interfaz, evitando ramificar lógica en el resto
  de la app.
- Contextvars `current_trace_id` y `current_observation_id` permiten que el
  AgentLoop y el retriever sepan dónde colgar sus spans sin tener que pasar
  el trace explícito por toda la pila.
- `litellm_metadata()` produce el dict que el `LLMClient` mete en
  `extra_body={"metadata": ...}` para que LiteLLM cree la generation bajo el
  trace correcto (integración nativa LiteLLM ↔ Langfuse).

Tolerancia a fallos: cualquier excepción del SDK se logea y se traga. Una
caída de Langfuse no debe romper el flujo del usuario.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

log = logging.getLogger(__name__)

# Contextvars accesibles desde cualquier punto del request (LLMClient, AgentLoop, ...)
current_trace_id: ContextVar[str | None] = ContextVar("current_trace_id", default=None)
current_observation_id: ContextVar[str | None] = ContextVar("current_observation_id", default=None)


class _SpanHandle:
    """Wrapper liviano sobre un span/generation de Langfuse.

    Maneja el end() final y absorbe excepciones del SDK para no propagarlas.
    """

    __slots__ = ("_obj", "_id", "_active")

    def __init__(self, obj: Any, obs_id: str | None):
        self._obj = obj
        self._id = obs_id
        self._active = True

    @property
    def id(self) -> str | None:
        return self._id

    def update(self, **kwargs: Any) -> None:
        if not self._active or self._obj is None:
            return
        try:
            self._obj.update(**kwargs)
        except Exception as e:  # pragma: no cover - el SDK puede fallar por red
            log.warning("langfuse.span_update_failed", extra={"err": str(e)})

    def end(self, **kwargs: Any) -> None:
        if not self._active:
            return
        self._active = False
        if self._obj is None:
            return
        try:
            self._obj.end(**kwargs)
        except Exception as e:  # pragma: no cover
            log.warning("langfuse.span_end_failed", extra={"err": str(e)})


class _NoopHandle(_SpanHandle):
    def __init__(self):  # type: ignore[no-untyped-def]
        super().__init__(None, None)


class LangfuseTracer:
    """Tracer real, envoltorio sobre el SDK de Langfuse."""

    def __init__(self, public_key: str, secret_key: str, host: str, log_payloads: bool):
        from langfuse import Langfuse  # type: ignore[import-not-found]

        self._lf = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            # El SDK loguea bastante en INFO. WARNING basta para nosotros.
            flush_at=20,
            flush_interval=5.0,
        )
        self.host = host
        self.log_payloads = log_payloads
        self.enabled = True

    # ---- traces ----

    def start_trace(
        self,
        name: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        input: Any | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> _SpanHandle:
        try:
            trace = self._lf.trace(
                name=name,
                user_id=user_id,
                session_id=session_id,
                input=input if self.log_payloads else None,
                metadata=metadata,
                tags=tags,
            )
        except Exception as e:
            log.warning("langfuse.trace_create_failed", extra={"err": str(e)})
            return _NoopHandle()
        return _SpanHandle(trace, getattr(trace, "id", None))

    @contextmanager
    def trace(
        self,
        name: str,
        **kwargs: Any,
    ) -> Iterator[_SpanHandle]:
        """Context manager: abre trace, setea contextvar, garantiza end()."""
        handle = self.start_trace(name, **kwargs)
        token = current_trace_id.set(handle.id)
        try:
            yield handle
        finally:
            handle.end()
            current_trace_id.reset(token)

    # ---- spans ----

    def start_span(
        self,
        name: str,
        *,
        input: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> _SpanHandle:
        trace_id = current_trace_id.get()
        if not trace_id:
            return _NoopHandle()
        parent = current_observation_id.get()
        try:
            span = self._lf.span(
                trace_id=trace_id,
                parent_observation_id=parent,
                name=name,
                input=input if self.log_payloads else None,
                metadata=metadata,
            )
        except Exception as e:
            log.warning("langfuse.span_create_failed", extra={"err": str(e)})
            return _NoopHandle()
        return _SpanHandle(span, getattr(span, "id", None))

    @contextmanager
    def span(
        self,
        name: str,
        **kwargs: Any,
    ) -> Iterator[_SpanHandle]:
        handle = self.start_span(name, **kwargs)
        token = current_observation_id.set(handle.id) if handle.id else None
        try:
            yield handle
        finally:
            handle.end()
            if token is not None:
                current_observation_id.reset(token)

    # ---- integración con LiteLLM ----

    def litellm_metadata(self, generation_name: str) -> dict[str, Any]:
        """Devuelve el bloque metadata para forwardear al proxy LiteLLM.

        LiteLLM crea automáticamente una generation en Langfuse cuando recibe
        este metadata en el request (con success_callback=langfuse activo en
        el proxy). La generation se cuelga del trace_id que pasamos.
        """
        trace_id = current_trace_id.get()
        if not trace_id:
            return {}
        meta: dict[str, Any] = {
            # `existing_trace_id` (y NO `trace_id` a secas) le dice al callback
            # Langfuse de LiteLLM que el trace YA existe y que solo debe anclar
            # la generation, sin reescribir campos del trace. Con `trace_id`
            # solo, LiteLLM hace upsert poniendo su nombre por defecto
            # (`litellm-acompletion`/`-aembedding`); como su flush compite con el
            # del SDK de la app sobre el mismo trace, el último en llegar gana y
            # acababa pisando el nombre real (`agent.run`, `chat.completions`).
            # Ver langfuse.py del proxy: con `existing_trace_id` hace
            # trace_params={"id": ...} sin `name`.
            "existing_trace_id": trace_id,
            "generation_name": generation_name,
        }
        parent = current_observation_id.get()
        if parent:
            meta["parent_observation_id"] = parent
        return meta

    # ---- lifecycle ----

    def shutdown(self) -> None:
        try:
            self._lf.flush()
            self._lf.shutdown()
        except Exception as e:  # pragma: no cover
            log.warning("langfuse.shutdown_failed", extra={"err": str(e)})


class NoopTracer:
    """Tracer dummy con la misma interfaz; usado cuando Langfuse está off."""

    enabled = False
    host = ""
    log_payloads = False

    def start_trace(self, *args: Any, **kwargs: Any) -> _SpanHandle:
        return _NoopHandle()

    @contextmanager
    def trace(self, name: str, **kwargs: Any) -> Iterator[_SpanHandle]:
        token = current_trace_id.set(None)
        try:
            yield _NoopHandle()
        finally:
            current_trace_id.reset(token)

    def start_span(self, *args: Any, **kwargs: Any) -> _SpanHandle:
        return _NoopHandle()

    @contextmanager
    def span(self, name: str, **kwargs: Any) -> Iterator[_SpanHandle]:
        yield _NoopHandle()

    def litellm_metadata(self, generation_name: str) -> dict[str, Any]:
        return {}

    def shutdown(self) -> None:
        return


_tracer: LangfuseTracer | NoopTracer = NoopTracer()


def init_tracer(
    *,
    enabled: bool,
    public_key: str,
    secret_key: str,
    host: str,
    log_payloads: bool,
) -> LangfuseTracer | NoopTracer:
    """Inicializa el singleton. Si faltan keys, queda en modo noop."""
    global _tracer
    if not enabled or not public_key or not secret_key:
        log.info("langfuse.disabled", extra={"enabled": enabled, "has_keys": bool(public_key and secret_key)})
        _tracer = NoopTracer()
        return _tracer
    try:
        _tracer = LangfuseTracer(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            log_payloads=log_payloads,
        )
        log.info("langfuse.ready", extra={"host": host})
    except Exception as e:
        log.warning("langfuse.init_failed", extra={"err": str(e)})
        _tracer = NoopTracer()
    return _tracer


def get_tracer() -> LangfuseTracer | NoopTracer:
    return _tracer


def shutdown_tracer() -> None:
    global _tracer
    try:
        _tracer.shutdown()
    finally:
        _tracer = NoopTracer()
