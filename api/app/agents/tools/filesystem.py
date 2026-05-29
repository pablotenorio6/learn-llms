"""Tools de filesystem confinadas a un directorio jail (sandbox).

Tres tools: fs_list, fs_read, fs_write. Todas operan SOLO dentro de
`settings.tools_sandbox_dir`. El confinamiento se hace resolviendo la ruta
(`Path.resolve()`, que también colapsa symlinks y `..`) y verificando que el
resultado sigue dentro del jail con `is_relative_to`. Cualquier intento de
salir (`../`, ruta absoluta, symlink que apunte fuera) se rechaza.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.agents.registry import tool
from app.config import get_settings

log = logging.getLogger(__name__)


class _JailError(Exception):
    pass


def _jail_root() -> Path:
    root = Path(get_settings().tools_sandbox_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_in_jail(rel_path: str) -> Path:
    """Resuelve `rel_path` dentro del jail o lanza _JailError si se sale."""
    root = _jail_root()
    # Tratamos toda ruta como relativa al jail; una absoluta se reinterpreta
    # como relativa (quitando la barra inicial) para que nunca escape.
    candidate = (root / rel_path.lstrip("/\\")).resolve()
    if candidate != root and root not in candidate.parents:
        raise _JailError(f"ruta fuera del sandbox: {rel_path!r}")
    return candidate


@tool
async def fs_list(path: str = ".") -> dict:
    """Lista archivos y carpetas dentro del workspace sandbox del agente. ÚSALA para ver qué hay disponible antes de leer o escribir. Las rutas son relativas a la raíz del sandbox; no puedes salir de él. Devuelve nombre, si es archivo o carpeta, y tamaño en bytes.

    Args:
        path: Subcarpeta a listar, relativa a la raíz del sandbox. Por defecto la raíz.
    """
    try:
        target = _resolve_in_jail(path)
    except _JailError as e:
        return {"error": str(e)}
    if not target.exists():
        return {"error": f"no existe: {path!r}"}
    if not target.is_dir():
        return {"error": f"no es una carpeta: {path!r}"}
    root = _jail_root()
    entries = []
    for child in sorted(target.iterdir()):
        entries.append({
            "name": child.name,
            "path": str(child.relative_to(root)).replace("\\", "/"),
            "type": "dir" if child.is_dir() else "file",
            "size": child.stat().st_size if child.is_file() else None,
        })
    return {"path": str(target.relative_to(root)).replace("\\", "/") or ".", "entries": entries}


@tool
async def fs_read(path: str) -> dict:
    """Lee el contenido de texto de un archivo del workspace sandbox del agente. ÚSALA para inspeccionar archivos que tú u otra tool habéis escrito, o que el usuario ha dejado en el sandbox. Solo texto (UTF-8); el contenido se trunca si es muy grande. No puede leer fuera del sandbox.

    Args:
        path: Ruta del archivo relativa a la raíz del sandbox.
    """
    settings = get_settings()
    try:
        target = _resolve_in_jail(path)
    except _JailError as e:
        return {"error": str(e)}
    if not target.exists() or not target.is_file():
        return {"error": f"no es un archivo existente: {path!r}"}
    raw = target.read_bytes()
    truncated = len(raw) > settings.fs_max_read_bytes
    if truncated:
        raw = raw[: settings.fs_max_read_bytes]
    text = raw.decode("utf-8", errors="replace")
    return {"path": path, "content": text, "bytes": len(raw), "truncated": truncated}


@tool
async def fs_write(path: str, content: str) -> dict:
    """Escribe (o sobrescribe) un archivo de texto en el workspace sandbox del agente. ÚSALA para guardar resultados intermedios, generar archivos que luego leerás o ejecutarás, o dejar output para el usuario. Crea las carpetas intermedias que hagan falta. No puede escribir fuera del sandbox.

    Args:
        path: Ruta destino relativa a la raíz del sandbox.
        content: Contenido de texto a escribir.
    """
    settings = get_settings()
    data = (content or "").encode("utf-8")
    if len(data) > settings.fs_max_write_bytes:
        return {"error": f"contenido demasiado grande ({len(data)} bytes > {settings.fs_max_write_bytes})"}
    try:
        target = _resolve_in_jail(path)
    except _JailError as e:
        return {"error": str(e)}
    if target.is_dir():
        return {"error": f"la ruta es una carpeta: {path!r}"}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return {"path": path, "bytes_written": len(data)}
