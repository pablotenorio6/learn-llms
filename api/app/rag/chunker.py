"""Partidor recursivo de texto en chunks con solape.

Estrategia: intenta partir por separadores de mayor a menor jerarquía
(párrafos → líneas → frases → palabras → caracteres) buscando trozos
de tamaño <= chunk_size. Es lo que hace `RecursiveCharacterTextSplitter`
de langchain pero en 60 líneas y sin dependencia.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


@dataclass
class Chunk:
    text: str
    index: int          # nº de orden dentro del documento
    char_start: int     # offset aproximado en el original
    char_end: int


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    separators: list[str] | None = None,
) -> list[Chunk]:
    """Devuelve una lista de Chunks de tamaño <= chunk_size con solape."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap debe ser < chunk_size")
    seps = separators or DEFAULT_SEPARATORS
    text = text.strip()
    if not text:
        return []

    splits = _recursive_split(text, chunk_size, seps)
    merged = _merge_with_overlap(splits, chunk_size, chunk_overlap)

    # Calcular offsets aproximados (búsqueda lineal: ok para tamaños razonables)
    out: list[Chunk] = []
    cursor = 0
    for i, t in enumerate(merged):
        idx = text.find(t, cursor)
        if idx == -1:
            idx = cursor
        out.append(Chunk(text=t, index=i, char_start=idx, char_end=idx + len(t)))
        cursor = idx + max(1, len(t) - chunk_overlap)
    return out


def _recursive_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Parte el texto recursivamente usando el primer separador que produce trozos pequeños."""
    if len(text) <= chunk_size:
        return [text]
    if not separators:
        # último recurso: corte duro por chunk_size
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep = separators[0]
    rest = separators[1:]
    if sep == "":
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(sep)
    out: list[str] = []
    for p in parts:
        # cada parte se vuelve a partir si sigue siendo grande
        if len(p) > chunk_size:
            out.extend(_recursive_split(p, chunk_size, rest))
        elif p:
            out.append(p)
    # Re-añadir el separador a los trozos para preservar contenido (excepto al último)
    if sep != "":
        out = [s + sep for s in out[:-1]] + (out[-1:] if out else [])
    return out


def _merge_with_overlap(splits: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Junta trozos pequeños hasta acercarse a chunk_size, con solape al final del anterior."""
    chunks: list[str] = []
    current = ""
    for s in splits:
        if len(current) + len(s) <= chunk_size:
            current += s
        else:
            if current:
                chunks.append(current.strip())
            # Empezar el siguiente con un solape del final del anterior
            if overlap > 0 and chunks:
                tail = chunks[-1][-overlap:]
                current = tail + s
            else:
                current = s
    if current.strip():
        chunks.append(current.strip())
    # Limpieza: eliminar duplicados consecutivos por si el solape produjo idénticos
    deduped: list[str] = []
    for c in chunks:
        if not deduped or c != deduped[-1]:
            deduped.append(c)
    return deduped
