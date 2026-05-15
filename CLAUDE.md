# llm_ops — Contexto para el siguiente agente

Proyecto de aprendizaje de despliegue/infraestructura de IA: asistente LLM local sobre Ollama + Qdrant, con harness propio (wrapper FastAPI OpenAI-compatible, RAG sobre Qdrant, agente con tools, UI vanilla).

**Hardware del usuario:** GTX 4060 8 GB VRAM, 32 GB DDR5 RAM, Windows con Docker Desktop.

**Idioma:** trabajamos en español. El usuario está aprendiendo infra de IA, no es junior — explicaciones técnicas con sustancia, sin condescender ni adornar. Prosa principalmente; listas cuando son comparaciones reales; evita formato pesado.

---

## Estado del roadmap

`roadmap.md` tiene el plan completo en 7 fases. Completadas:

- **Fase 0** — Setup baseline (Ollama, bench, modelos descargados)
- **Fase 1** — Wrapper API OpenAI-compatible (FastAPI + SSE streaming + cancelación + middleware de logging)
- **Fase 3** — RAG sobre Qdrant (Fase 2 se saltó deliberadamente para llegar antes a RAG)
- **Fase 4** — Tool calling + harness de agente (registry @tool, AgentLoop con SSE events, /v1/agents/run, UI con cards inline)
- **Multi-proveedor vía LiteLLM proxy** — la API ya no habla con Ollama directo. Habla con un proxy LiteLLM (contenedor sidecar) en formato OpenAI puro. Soporta Ollama local, OpenAI y Anthropic con sólo añadir alias en `litellm-config.yaml`

**Pendientes:**
- **Fase 2** — Observabilidad (Langfuse self-hosted para trazas LLM, Prometheus + Grafana para métricas de sistema)
- **Fase 5** — Evals + CI (Promptfoo o DeepEval, 50–100 ejemplos de regresión, integración en GitHub Actions, métricas RAG)
- **Fase 6** — Producción-grade (Redis caché, rate limiting, auth con API keys, router de modelos, load testing)
- **Fase 7** (opcional) — Frontend con Open WebUI o Next.js

**Mejoras a corto plazo de lo ya construido:**
- Reranker en el RAG (`bge-reranker-base`) — gran salto de calidad
- Búsqueda híbrida vector + BM25 (Qdrant lo soporta nativo desde 1.10)
- Más tools en el agente: filesystem sandboxed, http_fetch, calculadora, ejecución Python en subproceso aislado
- Pre-classifier antes del agent loop para suprimir tools cuando la pregunta es trivial
- Citas en línea: parsear `[source]` que el modelo emite y hacerlas clicables

---

## Stack actual

```
[Cliente browser] ──▶ [FastAPI :8000] ──▶ [LiteLLM :4000] ──┬──▶ Ollama :11434 (local, GPU)
                            │                                ├──▶ api.openai.com
                            └──▶ Qdrant :6333 (RAG)          └──▶ api.anthropic.com
```

Cuatro contenedores en la red interna de Docker (`docker-compose.yml`). Solo la API expone puerto al host (Qdrant también, para el dashboard). LiteLLM concentra el control plane: routing entre proveedores, gestión de keys, retries, futuro cost-tracking. La API habla OpenAI puro contra `http://litellm:4000/v1` con el master key del proxy. `docs/` y `bench/` están bind-mounted desde el host (iteras sin rebuild).

---

## Estructura

```
api/app/
  main.py              Entry point · lifespan (LLMClient+RAG+Watcher) · error handlers · ruta /
  config.py            Settings via pydantic-settings, leído de .env
  models.py            Schemas Pydantic: OpenAI + RAG + Agents
  middleware/
    logging.py         structlog JSON, X-Request-ID, timing por request
  services/
    llm_client.py      Cliente async contra LiteLLM proxy (AsyncOpenAI con
                       base_url=http://litellm:4000/v1). Métodos: chat,
                       chat_stream, chat_with_tools(_stream), embed, list_models
  routers/
    health.py          /healthz, /readyz (pinguea LiteLLM)
    chat.py            /v1/chat/completions (SSE)
    embeddings.py      /v1/embeddings
    models.py          /v1/models (listados por LiteLLM, aliases del config)
    rag.py             /v1/rag/documents (GET/POST/DELETE), /v1/rag/query, /v1/rag/reindex
    agents.py          /v1/agents/tools, /v1/agents/run (SSE de eventos del agente)
  rag/
    parsers.py         txt/md/pdf/html → texto plano (pypdf, beautifulsoup)
    chunker.py         Recursive character splitter, sin deps externas
    embedder.py        Wrapper sobre LLMClient.embed con batching
    store.py           QdrantStore async: ensure_collection, upsert_chunks, delete_by_*, search
    indexer.py         Orquesta parse→chunk→embed→upsert · idempotencia por sha256(chunk)
    retriever.py       query → embed → search → formatea como mensaje system
    watcher.py         watchdog en background task (carpeta ./docs)
  agents/
    registry.py        @tool decorator con schema autogen desde signature + docstring
    loop.py            AgentLoop async generator emitiendo eventos SSE + DEFAULT_AGENT_SYSTEM
                       Acumula tool_calls progresivos en streaming OpenAI.
    tools/
      __init__.py      Importa todas las tools (registra en REGISTRY al cargar)
      rag_search.py    Tool: búsqueda en la KB del usuario
      web.py           Tools: web_search, http_fetch
  static/
    index.html         UI vanilla (HTML/CSS/JS sin build): chat + RAG toggle +
                       Agente toggle + drag&drop + system prompt + dark mode + SSE
  tests/
    test_litellm_integration.py  E2E sobre TestClient + FakeLLM (mock de LLMClient)
bench/
  benchmark.py         TTFT, tokens/s vía LiteLLM proxy (AsyncOpenAI)
docs/                  Carpeta bind-mounted, vigilada por el watcher (RAG)
scripts/
  pull_models.sh       Descarga modelos baseline en Ollama
  smoke_test.sh        curl a todos los endpoints
docker-compose.yml     4 servicios: ollama (GPU), qdrant, litellm, api
litellm-config.yaml    Aliases de modelos del proxy (ollama_chat/, openai/, anthropic/)
.env.example
Makefile               up/down/logs/litellm-logs/rebuild/pull-models/bench/smoke/shell
roadmap.md             Plan completo
```

---

## Modelos usados

Los aliases viven en `litellm-config.yaml` (`model_name`). La API y la UI nunca usan el ID interno del proveedor: hablan el alias y el proxy enruta.

| Alias (proxy) | Backend real | VRAM/Coste | Notas |
|---|---|---:|---|
| `llama-local` | `ollama_chat/llama3.1:8b-instruct-q4_K_M` | ~4.9 GB | Buen tool calling pero **sesgo a over-calling** |
| `qwen-local` | `ollama_chat/qwen2.5:7b-instruct-q4_K_M` | ~4.7 GB | **Mejor disciplina** para tools, recomendado para modo agente |
| `phi-local` | `ollama/phi3.5:3.8b` | ~2.4 GB | No soporta tool calling nativo |
| `nomic-embed` | `ollama/nomic-embed-text` | ~280 MB | 768 dim, usado por el RAG |
| `gpt-4o-mini` | `openai/gpt-4o-mini` | $0.15/$0.60 por Mtok | Necesita `OPENAI_API_KEY` |
| `gpt-4o` | `openai/gpt-4o` | $2.50/$10 por Mtok | |
| `claude-sonnet-4-6` | `anthropic/claude-sonnet-4-6` | $3/$15 por Mtok | Necesita `ANTHROPIC_API_KEY` |
| `claude-haiku-4-5` | `anthropic/claude-haiku-4-5-20251001` | $1/$5 por Mtok | |

`make pull-models` descarga los locales. Para añadir un proveedor nuevo basta con agregar una entrada al `model_list` del YAML y reiniciar el contenedor `litellm` — la API no necesita rebuild.

---

## Cómo arrancar

```powershell
cp .env.example .env
# Rellena OPENAI_API_KEY y/o ANTHROPIC_API_KEY si quieres usar cloud.
# LITELLM_MASTER_KEY puede dejarse el default en local.
docker compose up -d --build
make pull-models           # solo la primera vez (~12 GB de descarga)
make smoke                 # opcional, sanity check
```

UI en `http://localhost:8000/`, Qdrant dashboard en `http://localhost:6333/dashboard`.

---

## Comandos habituales

```bash
make up / down / rebuild / logs
make pull-models           # descarga modelos
make bench                 # benchmark sobre BENCH_MODELS
make smoke                 # curl a todos los endpoints
make shell                 # bash dentro del contenedor api
```

---

## Convenciones

- **Imports** absolutos desde el paquete (`from app.X import Y`). En testing: `PYTHONPATH=api`.
- **Async everywhere** — funciones de I/O son async. El loop del agente es un async generator.
- **Pydantic v2** para todos los schemas. `Field(...)` con `ge`/`le` donde aplique.
- **structlog** para logs JSON. Cada request lleva `X-Request-ID` propagado vía `contextvars`.
- **SSE format** estilo OpenAI: `data: {json}\n\n` separados por línea en blanco, terminado en `data: [DONE]\n\n`.
- **Tool calls** estilo OpenAI: `{"type":"function","function":{"name":..., "parameters":...}}`. El decorador `@tool` lo autogenera desde la signature inspeccionada con Pydantic `create_model`.
- **AgentLoop eventos**: `iteration` · `content_delta` · `tool_call` · `tool_result` · `tool_error` · `final` · `limit_reached` · `error`.

---

## Gotchas aprendidos por las malas

1. **Edit en docs largos puede truncar** silenciosamente en mi entorno. Si haces cambios masivos en archivos como `index.html` o `loop.py`, valida tamaño antes/después o reescribe con `cat > file <<'EOF'`.

2. **Heredocs Python ↔ escape JS:** los `\\n\\n` en `str.replace()` se desescapan a saltos de línea reales y rompen JS literal. Si embebes JS en el HTML mediante Python heredoc, valida con `node -c <(extract.py)`.

3. **Healthcheck Docker Compose:** preferir `CMD` (lista) sobre `CMD-SHELL` con comillas escapadas — PyYAML es más estricto que el parser de Docker Compose y se atraganta con `\"` aunque Docker funciona.

4. **bench/ y docs/ son bind-mounts** (en `docker-compose.yml`). La imagen solo copia `app/`. Editar `benchmark.py` no requiere rebuild; editar código de `api/app/` sí.

5. **LLMClient habla con LiteLLM, no con Ollama directo.** La traducción al dialecto de cada proveedor la hace el proxy (los aliases `ollama_chat/...` activan el modo de tool-calling estructurado de Ollama). Si añades un proveedor nuevo, lo configuras en `litellm-config.yaml` y la API no necesita cambios.

6. **Tool calling streaming OpenAI:** los `tool_calls` llegan como **deltas progresivos** con `index` estable y `function.arguments` concatenándose chunk a chunk. El `AgentLoop` acumula por `index` y cierra por `finish_reason == "tool_calls"`. Esto difiere del comportamiento legacy de Ollama (todo en el chunk final con `done: true`); ahora pasa por LiteLLM y se normaliza al formato OpenAI puro. Tools paralelas (varias en una misma respuesta) ya están soportadas.

7. **RAG idempotencia:** el `point_id` en Qdrant es `uuid5(NAMESPACE_OID, sha256(chunk_text))`. Re-indexar el mismo doc no duplica. Antes de `upsert` se llama a `delete_by_source(source)` para limpiar chunks viejos si el contenido cambió.

8. **Default `num_ctx` de Ollama** suele ser 2048. Para conversaciones largas con RAG (que inyecta chunks) puedes querer subirlo vía `options.num_ctx` en la llamada — no está expuesto aún en el wrapper.

9. **Prompt engineering para tools en 7-8B es iterativo y específico del modelo:**
   - Llama 3.1 8B → **over-calling** (recall alto, precisión baja)
   - Qwen 2.5 7B → **under-calling** y tiende a narrar su decisión al usuario ("voy a llamar X…")
   - El `DEFAULT_AGENT_SYSTEM` en `agents/loop.py` está calibrado tras varias iteraciones para minimizar ambos sesgos: incluye sinónimos de la KB ("base de datos vectorial", "RAG", "mis notas"), instrucciones explícitas de "no narres tu decisión", clarificación "documentos del usuario ≠ PII protegida". Si lo tocas, valida con:
     - Trivial: *"¿de qué color es una manzana?"* → no debería llamar
     - Clara: *"¿qué dice mi PDF de X?"* → debería llamar
     - Ambigua: *"qué hay en la base de datos"* → debería llamar (regla "ante la duda, llama")

10. **El sandbox del usuario tiene proxies SOCKS** activos por defecto que rompen httpx en tests. Si tropiezas con `ImportError: socksio not installed`, `unset HTTP_PROXY HTTPS_PROXY ALL_PROXY ftp_proxy grpc_proxy all_proxy` antes de los tests.

---

## Testing

Patrón habitual en sandbox:

```bash
# Copiar el código fuera del fs read-only y bypassar pycache stale
rm -rf /tmp/api_test && mkdir -p /tmp/api_test
cp -r api/app /tmp/api_test/

# Venv con deps
python3 -m venv /tmp/v
/tmp/v/bin/pip install -q -r api/requirements.txt

# Tests con TestClient apuntando a hosts inexistentes (mocks por 502/503)
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY FTP_PROXY GRPC_PROXY RSYNC_PROXY ...
cd /tmp/api_test
OLLAMA_HOST=http://x:1 QDRANT_HOST=http://x:6333 RAG_WATCHER_ENABLED=false \
  PYTHONPATH=. /tmp/v/bin/python -B test_script.py
```

Para mockear el cliente dentro del lifespan, sustituir `app.state.llm` con un fake **dentro** del `with TestClient(app)` block (el lifespan startup sobreescribe lo que pongas antes). Hay un ejemplo trabajado en `api/tests/test_litellm_integration.py` con un `FakeLLM` que cubre chat no-stream, streaming, agente trivial, 1 tool y tools paralelas.

---

## Notas sobre la conversación contigo

Hemos hecho 4 fases con un patrón estable: el usuario pregunta o pide una fase, yo propongo AskUserQuestion con 2–3 decisiones de diseño clave, él elige, yo creo tasks, voy ejecutando con verificación end-to-end por TestClient + mocks, y resumo al final.

Para preguntas conceptuales (cómo funciona X), prosa larga y de profundidad. Para bugs reportados, diagnóstico → fix → verificación → resumen breve. El usuario aprende mejor cuando explico el "por qué" después del "qué".

Última iteración: hubo un bug donde una sustitución regex en `index.html` rompió el JS por mal escape de `\n\n`. Si el frontend muestra "conectando…" eterno y el dropdown "cargando…", lo primero es validar el JS con `node -c` sobre el `<script>` extraído.

---

## Ficheros clave que conviene leer primero

1. `roadmap.md` — el plan global y dónde estamos
2. `api/app/main.py` — entry point y wiring de todo
3. `api/app/agents/loop.py` — el bucle del agente y `DEFAULT_AGENT_SYSTEM`
4. `api/app/rag/indexer.py` + `retriever.py` — flujo RAG
5. `api/app/static/index.html` — UI completa, sirve también de "manual de uso" del API
6. `docker-compose.yml` — orquestación y healthchecks
