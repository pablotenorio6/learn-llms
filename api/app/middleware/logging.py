"""Logging estructurado en JSON + correlation IDs por request.

Uso:
- `configure_logging(level)` al arranque (lifespan).
- `RequestContextMiddleware` añade un X-Request-ID a la respuesta y lo inyecta
  en el contexto de structlog para que aparezca en todos los logs del request.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextvars import ContextVar

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

_request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def _add_request_id(_, __, event_dict):
    event_dict["request_id"] = _request_id_var.get()
    return event_dict


def configure_logging(level: str = "INFO") -> None:
    timestamper = structlog.processors.TimeStamper(fmt="iso")
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
        _add_request_id,
    ]

    structlog.configure(
        processors=shared_processors
        + [structlog.processors.JSONRenderer()],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )

    # Hacer que el logging stdlib también salga JSON
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=shared_processors,
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Bajar el ruido de uvicorn.access (lo emitimos nosotros desde el middleware)
    logging.getLogger("uvicorn.access").disabled = True


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        token = _request_id_var.set(rid)
        log = structlog.get_logger("api.request")
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["x-request-id"] = rid
            return response
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            log.info(
                "http_request",
                method=request.method,
                path=request.url.path,
                status=status_code,
                duration_ms=round(elapsed_ms, 2),
                client=request.client.host if request.client else None,
            )
            _request_id_var.reset(token)
