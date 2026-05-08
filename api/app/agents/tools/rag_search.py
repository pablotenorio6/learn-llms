"""Tool: búsqueda en la base de conocimiento del usuario (Qdrant)."""

from __future__ import annotations

from app.agents.registry import tool


@tool
async def rag_search(ctx, query: str, top_k: int = 5) -> dict:
    """Busca en la BASE DE CONOCIMIENTO PRIVADA del usuario — también llamada 'base de datos vectorial', 'vector DB', 'Qdrant', 'RAG', 'índice', 'mis documentos', 'mis notas', 'mis archivos', 'mis PDFs', 'la KB'. Devuelve los chunks más relevantes para responder. ÚSALA cuando la pregunta haga referencia (clara o plausible) a contenido del usuario: sus apuntes, archivos, PDFs subidos, notas personales, o cuando él mencione explícitamente "base de datos", "RAG", "mis documentos" o sinónimos. Ante la duda razonable, llámala — si vuelve vacía dirás que no encontraste y responderás con conocimiento general. NO la uses para conocimiento general claramente público (geografía, ciencia básica, definiciones), saludos, conversación trivial, matemáticas u opiniones. Ejemplos: SÍ → "qué dicen mis notas sobre OAuth", "resume el PDF que subí", "busca en la base de datos info sobre X", "lo tengo en mi RAG". NO → "de qué color es una manzana", "hola", "cuánto es 2+2", "escríbeme un poema".

    Args:
        query: Pregunta o términos clave en lenguaje natural sobre los que buscar en los documentos.
        top_k: Número de chunks a recuperar (1-10). Por defecto 5.
    """
    if ctx is None or ctx.get("rag") is None:
        return {"error": "RAG no inicializado en este servidor", "hits": []}
    top_k = max(1, min(10, int(top_k)))
    res = await ctx["rag"]["retriever"].query(query, top_k=top_k)
    return {
        "query": res.query,
        "n_hits": len(res.hits),
        "hits": [
            {
                "source": h.source,
                "score": round(h.score, 4),
                "chunk_idx": h.chunk_idx,
                "text": h.text,
            }
            for h in res.hits
        ],
    }
