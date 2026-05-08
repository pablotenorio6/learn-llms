"""Tool: búsqueda en la base de conocimiento del usuario (Qdrant)."""

from __future__ import annotations

from app.agents.registry import tool


@tool
async def rag_search(ctx, query: str, top_k: int = 5) -> dict:
    """Busca en la base de conocimiento del usuario (documentos indexados en Qdrant) y devuelve los chunks más relevantes para responder. Úsala cuando la pregunta pueda ser respondida con documentos privados del usuario.

    Args:
        query: Pregunta o términos clave en lenguaje natural sobre los que buscar.
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
