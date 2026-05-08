"""Conversión de archivos a texto plano por extensión.

Mantengo cada parser pequeño y explícito para que sea fácil añadir formatos.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf", ".html", ".htm"}


class UnsupportedFormatError(Exception):
    pass


def parse_file(path: Path) -> str:
    """Devuelve texto plano del archivo. Lanza UnsupportedFormatError si no se sabe parsear."""
    ext = path.suffix.lower()
    if ext in {".txt", ".md", ".markdown"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if ext == ".pdf":
        return _parse_pdf(path)
    if ext in {".html", ".htm"}:
        return _parse_html(path)
    raise UnsupportedFormatError(f"extensión no soportada: {ext}")


def parse_bytes(data: bytes, filename: str) -> str:
    """Versión que opera sobre bytes (uploads). Decide por extensión del filename."""
    ext = Path(filename).suffix.lower()
    if ext in {".txt", ".md", ".markdown"}:
        return data.decode("utf-8", errors="replace")
    if ext == ".pdf":
        return _parse_pdf_bytes(data)
    if ext in {".html", ".htm"}:
        return _parse_html_bytes(data)
    raise UnsupportedFormatError(f"extensión no soportada: {ext}")


def _parse_pdf(path: Path) -> str:
    return _parse_pdf_bytes(path.read_bytes())


def _parse_pdf_bytes(data: bytes) -> str:
    from io import BytesIO
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))
    parts: list[str] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            log.warning("pdf.extract_failed", extra={"page": i, "err": str(e)})
            text = ""
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts)


def _parse_html(path: Path) -> str:
    return _parse_html_bytes(path.read_bytes())


def _parse_html_bytes(data: bytes) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(data, "html.parser")
    # Quita script/style ruidoso
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Colapsa líneas vacías repetidas
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)
