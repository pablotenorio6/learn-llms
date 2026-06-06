"""Conversión de archivos a texto plano por extensión.

Mantengo cada parser pequeño y explícito para que sea fácil añadir formatos.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf", ".html", ".htm"}

# Control chars salvo \n y \t. Aparecen p.ej. como glyphs de fuentes de iconos
# en CVs con plantilla (teléfono/email) y son ruido para embeddings y BM25.
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MANY_BLANKS_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normaliza Unicode (NFC), elimina control chars y colapsa espacios.

    Conservador: no toca el contenido, solo el ruido de extracción que degrada
    tanto el embedding como el match léxico.
    """
    text = unicodedata.normalize("NFC", text)
    text = _CTRL_RE.sub(" ", text)
    # espacios/tabs repetidos → uno; trailing por línea; 3+ saltos → 2
    text = "\n".join(re.sub(r"[ \t]{2,}", " ", ln).rstrip() for ln in text.splitlines())
    text = _MANY_BLANKS_RE.sub("\n\n", text)
    return text.strip()


class UnsupportedFormatError(Exception):
    pass


def parse_file(path: Path) -> str:
    """Devuelve texto plano del archivo. Lanza UnsupportedFormatError si no se sabe parsear."""
    ext = path.suffix.lower()
    if ext in {".txt", ".md", ".markdown"}:
        return clean_text(path.read_text(encoding="utf-8", errors="replace"))
    if ext == ".pdf":
        return _parse_pdf(path)
    if ext in {".html", ".htm"}:
        return _parse_html(path)
    raise UnsupportedFormatError(f"extensión no soportada: {ext}")


def parse_bytes(data: bytes, filename: str) -> str:
    """Versión que opera sobre bytes (uploads). Decide por extensión del filename."""
    ext = Path(filename).suffix.lower()
    if ext in {".txt", ".md", ".markdown"}:
        return clean_text(data.decode("utf-8", errors="replace"))
    if ext == ".pdf":
        return _parse_pdf_bytes(data)
    if ext in {".html", ".htm"}:
        return _parse_html_bytes(data)
    raise UnsupportedFormatError(f"extensión no soportada: {ext}")


def _parse_pdf(path: Path) -> str:
    return _parse_pdf_bytes(path.read_bytes())


def _parse_pdf_bytes(data: bytes) -> str:
    # PyMuPDF (fitz) extrae con mejor fidelidad de espaciado/encoding que pypdf,
    # sobre todo en PDFs con fuentes sin ToUnicode bien formado.
    import pymupdf

    parts: list[str] = []
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            try:
                text = page.get_text() or ""
            except Exception as e:
                log.warning("pdf.extract_failed", extra={"page": i, "err": str(e)})
                text = ""
            if text.strip():
                parts.append(text)
    return clean_text("\n\n".join(parts))


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
    return clean_text("\n".join(ln for ln in lines if ln))
