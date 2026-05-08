"""Registry de tools + decorador @tool con schema autogen.

El decorador inspecciona la signature y la docstring de la función
para construir un schema OpenAI-compatible (`type=function`).

Convenciones:
- Si la primera signatura es `ctx`, no aparece en el schema y la inyecta el AgentLoop.
- La descripción de la tool sale de la primera línea de la docstring.
- Las descripciones de cada parámetro se sacan de un bloque "Args:" si existe.
- El tipo viene del type hint; los defaults marcan parámetros opcionales.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, get_type_hints

from pydantic import Field, ValidationError, create_model

log = logging.getLogger(__name__)

ToolFn = Callable[..., Awaitable[Any]]


@dataclass
class Tool:
    name: str
    description: str
    fn: ToolFn
    schema: dict           # OpenAI-compatible: {"type":"function", "function": {...}}
    param_model: type      # Pydantic model para validar args
    needs_ctx: bool        # si True, llamamos fn(ctx, **args), si no fn(**args)

    async def call(self, ctx: Any, raw_args: dict) -> Any:
        try:
            validated = self.param_model.model_validate(raw_args)
        except ValidationError as e:
            raise ToolArgsError(f"args inválidos: {e.errors()}") from e
        kwargs = validated.model_dump()
        if self.needs_ctx:
            return await self.fn(ctx, **kwargs)
        return await self.fn(**kwargs)


class ToolArgsError(Exception):
    pass


REGISTRY: dict[str, Tool] = {}


def get_tools(allow: list[str] | None = None) -> list[Tool]:
    """Devuelve las tools del registry, opcionalmente filtradas por allow-list."""
    if allow is None:
        return list(REGISTRY.values())
    return [REGISTRY[name] for name in allow if name in REGISTRY]


def tool(_fn: ToolFn | None = None, *, name: str | None = None, description: str | None = None) -> ToolFn:
    """Decorador. Registra la función en REGISTRY y devuelve la función intacta."""

    def wrap(fn: ToolFn) -> ToolFn:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"@tool requiere async def: {fn.__name__}")
        sig = inspect.signature(fn)
        hints = get_type_hints(fn)

        params = list(sig.parameters.items())
        needs_ctx = bool(params) and params[0][0] in ("ctx", "context")
        if needs_ctx:
            params = params[1:]

        param_descriptions = _parse_arg_descriptions(fn.__doc__ or "")
        fields: dict[str, Any] = {}
        for pname, p in params:
            ann = hints.get(pname, str)
            descr = param_descriptions.get(pname)
            if p.default is inspect.Parameter.empty:
                fields[pname] = (ann, Field(..., description=descr))
            else:
                fields[pname] = (ann, Field(default=p.default, description=descr))

        param_model = create_model(fn.__name__ + "Args", **fields)  # type: ignore[arg-type]
        json_schema = param_model.model_json_schema()

        # Limpiar refs, no las necesitamos para tipos básicos
        params_schema = {
            "type": "object",
            "properties": json_schema.get("properties", {}),
            "required": json_schema.get("required", []),
        }

        tool_name = name or fn.__name__
        first_line = (fn.__doc__ or "").strip().split("\n", 1)[0].strip() if fn.__doc__ else ""
        tool_descr = description or first_line or tool_name

        schema = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": tool_descr,
                "parameters": params_schema,
            },
        }

        t = Tool(
            name=tool_name,
            description=tool_descr,
            fn=fn,
            schema=schema,
            param_model=param_model,
            needs_ctx=needs_ctx,
        )
        REGISTRY[tool_name] = t
        log.info("tool.registered", extra={"name": tool_name, "needs_ctx": needs_ctx})
        return fn

    if _fn is not None:
        return wrap(_fn)
    return wrap


def _parse_arg_descriptions(docstring: str) -> dict[str, str]:
    """Extrae descripciones por argumento del bloque 'Args:' al estilo Google."""
    out: dict[str, str] = {}
    m = re.search(r"\n\s*Args:\s*\n(.*?)(?:\n\s*(?:Returns|Raises|Yields):|\Z)", docstring, re.S)
    if not m:
        return out
    block = m.group(1)
    for line in block.splitlines():
        m2 = re.match(r"\s+(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)", line)
        if m2:
            out[m2.group(1)] = m2.group(2).strip()
    return out


def serialize_for_llm(value: Any, max_chars: int = 4000) -> str:
    """Serializa un resultado de tool para meterlo en el siguiente prompt.

    Trunca a max_chars para evitar reventar el contexto con resultados gigantes.
    """
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(value)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... [truncado: {len(text) - max_chars} chars]"
    return text
