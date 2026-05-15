"""Tools de web: búsqueda (Brave) y descarga/extracción de páginas.

Dos tools:
- web_search(query, count): busca en Brave Search API y devuelve [{title, url, snippet}].
- http_fetch(url): descarga la página, valida que no sea SSRF, limpia el HTML y devuelve texto.

Settings vienen de Settings (config.py): BRAVE_API_KEY, timeouts, etc.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
import socket
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from app.agents.registry import tool
from app.config import get_settings

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# web_search (Brave)
# ---------------------------------------------------------------------------

@tool
async def web_search(query: str, count: int = 5) -> dict:
    """Busca en la WEB PÚBLICA vía Brave Search. Devuelve una lista de resultados con título, URL y snippet. ÚSALA cuando la pregunta requiera información actual, noticias, hechos verificables que no conoces o que pueden haber cambiado, documentación de software, precios, eventos recientes, o cualquier dato que no puedas conocer por entrenamiento general. NO la uses para conocimiento estable y básico (matemáticas, geografía elemental, definiciones comunes), saludos, opiniones, escritura creativa, o cuando la pregunta sea sobre los documentos del usuario (para eso usa rag_search). Si los snippets no son suficientes, encadena con http_fetch sobre la URL más prometedora para leer la página completa.

    Args:
        query: Términos de búsqueda en lenguaje natural.
        count: Número de resultados a devolver (1-10). Por defecto 5.
    """
    settings = get_settings()
    if not settings.brave_api_key:
        return {
            "error": "BRAVE_API_KEY no configurada en el servidor",
            "results": [],
        }

    count = max(1, min(settings.web_search_max_results, int(count)))
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": settings.brave_api_key,
    }
    params = {"q": query, "count": count}

    try:
        async with httpx.AsyncClient(timeout=settings.web_search_timeout) as client:
            r = await client.get(settings.brave_endpoint, headers=headers, params=params)
    except httpx.HTTPError as e:
        log.warning("web_search.http_error", extra={"err": str(e)})
        return {"error": f"fallo de red contra Brave: {e}", "results": []}

    if r.status_code == 401 or r.status_code == 403:
        return {"error": "API key de Brave inválida o sin permisos", "results": []}
    if r.status_code == 429:
        return {"error": "rate limit de Brave alcanzado", "results": []}
    if r.status_code >= 400:
        return {"error": f"Brave devolvió {r.status_code}", "results": []}

    try:
        data = r.json()
    except ValueError:
        return {"error": "respuesta de Brave no es JSON", "results": []}

    web = (data.get("web") or {}).get("results") or []
    results = []
    for item in web[:count]:
        results.append({
            "title": (item.get("title") or "").strip(),
            "url": item.get("url") or "",
            "snippet": _clean_html_inline(item.get("description") or ""),
        })

    return {
        "query": query,
        "n": len(results),
        "results": results
    }


# ---------------------------------------------------------------------------
# http_fetch (con SSRF guard + extracción de texto)
# ---------------------------------------------------------------------------

@tool
async def http_fetch(url: str) -> dict:
    """Descarga una página HTTP/HTTPS pública y devuelve su texto limpio. Úsala como SEGUNDO PASO tras web_search cuando un snippet no basta para responder y necesitas leer el contenido completo de una URL específica (un artículo, una doc, una página de noticias). Bloquea automáticamente URLs internas/privadas (localhost, redes RFC1918, metadata cloud) por seguridad. NO la uses para descargar archivos binarios (PDFs, imágenes, vídeos): solo páginas HTML o texto plano. El contenido devuelto está truncado a unos miles de caracteres — si la página es muy larga, lee lo que devuelve y razona con eso.

    Args:
        url: URL completa con esquema (http:// o https://). Ej: https://example.com/articulo
    """
    settings = get_settings()

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return {"error": f"esquema no permitido: {parsed.scheme!r} (solo http/https)", "url": url}
    if not parsed.hostname:
        return {"error": "URL sin hostname", "url": url}

    safe, reason = await _is_public_host(parsed.hostname)
    if not safe:
        return {"error": f"URL bloqueada por seguridad: {reason}", "url": url}

    headers = {
        "User-Agent": settings.http_fetch_user_agent,
        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
        "Accept-Language": "es,en;q=0.8",
    }

    try:
        async with httpx.AsyncClient(
            timeout=settings.http_fetch_timeout,
            follow_redirects=True,
            max_redirects=5,
        ) as client:
            async with client.stream("GET", url, headers=headers) as r:
                final_host = (r.url.host or "")
                # Revalidar host final tras redirecciones
                safe2, reason2 = await _is_public_host(final_host)
                if not safe2:
                    return {
                        "error": f"redirección a host bloqueado: {final_host} ({reason2})",
                        "url": url,
                    }

                ctype = (r.headers.get("content-type") or "").lower()
                if not (ctype.startswith("text/") or "xml" in ctype or "json" in ctype):
                    return {
                        "error": f"content-type no soportado: {ctype!r}",
                        "url": str(r.url),
                        "status": r.status_code,
                    }

                buf = bytearray()
                async for chunk in r.aiter_bytes():
                    buf.extend(chunk)
                    if len(buf) >= settings.http_fetch_max_bytes:
                        break

                status = r.status_code
                final_url = str(r.url)
                encoding = r.encoding or "utf-8"
    except httpx.HTTPError as e:
        log.warning("http_fetch.http_error", extra={"err": str(e), "url": url})
        return {"error": f"fallo de red: {e}", "url": url}

    try:
        raw = bytes(buf).decode(encoding, errors="replace")
    except LookupError:
        raw = bytes(buf).decode("utf-8", errors="replace")

    if "html" in ctype:
        title, text = _extract_text_from_html(raw)
    else:
        title, text = "", raw

    truncated = False
    if len(text) > settings.http_fetch_max_chars:
        text = text[: settings.http_fetch_max_chars]
        truncated = True

    return {
        "url": final_url,
        "status": status,
        "content_type": ctype,
        "title": title,
        "text": text,
        "length": len(text),
        "truncated": truncated,
    }


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

_BLOCKED_TLDS = {"local", "internal", "localhost"}


async def _is_public_host(host: str) -> tuple[bool, str]:
    """Devuelve (True, "") si el host es público; (False, motivo) si no."""
    if not host:
        return False, "host vacío"
    host_l = host.lower().rstrip(".")
    if host_l in {"localhost"}:
        return False, "localhost"
    if any(host_l.endswith("." + s) or host_l == s for s in _BLOCKED_TLDS):
        return False, f"TLD interno ({host_l})"

    # Resolver todas las IPs del host (puede ser literal o nombre)
    try:
        loop = asyncio.get_running_loop()
        infos = await loop.getaddrinfo(host_l, None, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        return False, f"no resuelve DNS: {e}"

    for info in infos:
        sockaddr = info[4]
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            return False, f"IP inválida resuelta: {ip_str}"
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            return False, f"IP no pública ({ip})"
        # AWS/GCP metadata endpoint
        if str(ip) == "169.254.169.254":
            return False, "endpoint de metadata cloud"

    return True, ""


_WS_RE = re.compile(r"[ \t]+")
_NL_RE = re.compile(r"\n{3,}")


def _extract_text_from_html(html: str) -> tuple[str, str]:
    """Devuelve (title, text) extraído del HTML con bs4."""
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "form", "aside", "iframe", "svg"]):
        tag.decompose()

    # Preferir <article>, luego <main>, luego <body>
    root = soup.find("article") or soup.find("main") or soup.body or soup
    text = root.get_text(separator="\n", strip=True)

    # Limpiar whitespace
    text = _WS_RE.sub(" ", text)
    text = _NL_RE.sub("\n\n", text)
    return title, text.strip()


def _clean_html_inline(s: str) -> str:
    """Quita tags de un snippet pequeño (los snippets de Brave traen <strong>...)."""
    if not s:
        return ""
    return BeautifulSoup(s, "html.parser").get_text(" ", strip=True)
