# llm_ops — Contexto para el siguiente agente

Proyecto de aprendizaje de despliegue/infraestructura de IA: asistente LLM local sobre Ollama + Qdrant, con harness propio (wrapper FastAPI OpenAI-compatible, RAG sobre Qdrant, agente con tools, UI vanilla).

**Hardware del usuario:** GTX 4060 8 GB VRAM, 32 GB DDR5 RAM, Windows con Docker Desktop.

**Idioma:** trabajamos en español. El usuario está aprendiendo infra de IA, no es junior — explicaciones técnicas con sustancia, sin condescender ni adornar. Prosa principalmente; listas cuando son comparaciones reales; evita formato pesado.

---

## Estado del roadmap

`roadmap.md` tiene el plan completo en 7 fases. Completadas:

- **Fase 0** — Setup baseline (Ollama, bench, modelos descargados)
- **Fase 1** — Wrapper API OpenAI-compatible (FastAPI + SSE streaming + cancelación + middleware de logging)
- **Fase 2** — Observabilidad: Langfuse v2 self-hosted (trazas LLM con jerarquía trace→span→generation, integrado con LiteLLM vía callback + metadata.trace_id), Prometheus (/metrics en API + LiteLLM + side-car GPU con nvidia-smi), Grafana con dashboard provisionado
- **Fase 3** — RAG sobre Qdrant
- **Fase 4** — Tool calling + harness de agente (registry @tool, AgentLoop con SSE events, /v1/agents/run, UI con cards inline)
- **Multi-proveedor vía LiteLLM proxy** — la API ya no habla con Ollama directo. Habla con un proxy LiteLLM (contenedor sidecar) en formato OpenAI puro. Soporta Ollama local, OpenAI y Anthropic con sólo añadir alias en `litellm-config.yaml`

**Pendientes:**
- **Fase 5** — Evals + CI (Promptfoo o DeepEval, 50–100 ejemplos de regresión, integración en GitHub Actions, métricas RAG)
- **Fase 6** — Producción-grade (Redis caché, rate limiting, auth con API keys, router de modelos, load testing)
- **Fase 7** (opcional) — Frontend con Open WebUI o Next.js

**Mejoras a corto plazo de lo ya construido:**
- Reranker en el RAG (`bge-reranker-base`) — gran salto de calidad
- ~~Búsqueda híbrida vector + BM25~~ **HECHO** (app-side: BM25 en memoria + fusión RRF en `retriever.py`, ver gotcha #21). Pendiente opcional: migrar a sparse vectors nativos de Qdrant cuando el corpus crezca.
- Más tools en el agente: filesystem sandboxed, http_fetch, calculadora, ejecución Python en subproceso aislado
- Pre-classifier antes del agent loop para suprimir tools cuando la pregunta es trivial
- Citas en línea: parsear `[source]` que el modelo emite y hacerlas clicables
- ~~Optimización de latencia~~ **HECHO** (ver gotcha #23): warm-up de modelos al arrancar, tool calls paralelas en el AgentLoop, ramas denso/BM25 del retriever solapadas, LRU de embeddings de query, cache TTL de `/v1/models`, throttle de `is_disconnected()`, `OLLAMA_KEEP_ALIVE` 30m.

---

## Stack actual

```
[Cliente browser] ──▶ [FastAPI :8000] ──▶ [LiteLLM :4000] ──┬──▶ Ollama :11434 (local, GPU)
                            │                                ├──▶ api.openai.com
                            │                                └──▶ api.anthropic.com
                            ├──▶ Qdrant :6333 (RAG)
                            └──▶ Langfuse :3030 (trazas)

Observabilidad: Prometheus :9090 raspa /metrics de api + litellm + gpu-exporter,
Grafana :3001 muestra el dashboard provisionado.
```

Stack completo en `docker-compose.yml` (9 servicios). Puertos al host: API `8000`, Qdrant `6333`, LiteLLM `4000`, Langfuse `3030`, Prometheus `9090`, Grafana `3001`, gpu-exporter `9835`. LiteLLM concentra el control plane: routing, retries, cost tracking, callbacks Langfuse+Prometheus. La API habla OpenAI puro contra `http://litellm:4000/v1` con el master key del proxy. `docs/` y `bench/` están bind-mounted desde el host (iteras sin rebuild). `make urls` imprime todas las URLs útiles.

---

## Estructura

```
api/app/
  main.py              Entry point · lifespan (LLMClient+RAG+Watcher+Langfuse) ·
                       error handlers · expone /metrics vía Instrumentator
  config.py            Settings via pydantic-settings, leído de .env
  models.py            Schemas Pydantic: OpenAI + RAG + Agents
  middleware/
    logging.py         structlog JSON, X-Request-ID, timing por request
  observability/
    langfuse_client.py Tracer singleton (LangfuseTracer / NoopTracer) con
                       contextvars current_trace_id/current_observation_id.
                       Helpers: trace(), span() context managers + start_*().
                       litellm_metadata() produce el dict que viaja en
                       extra_body.metadata a LiteLLM para que la generation
                       cuelgue del trace correcto.
    metrics.py         Histogramas/counters/gauges Prometheus LLM-específicos:
                       llm_request_duration_seconds, llm_ttft_seconds,
                       llm_tokens_total, llm_active_requests,
                       agent_iterations_total, agent_tool_calls_total,
                       agent_tool_duration_seconds, rag_*.
  services/
    llm_client.py      Cliente async contra LiteLLM proxy (AsyncOpenAI con
                       base_url=http://litellm:4000/v1). Métodos: chat,
                       chat_stream, chat_with_tools(_stream), embed, list_models.
                       Cada método inyecta metadata Langfuse en extra_body y
                       emite métricas (TTFT, duración, tokens, active).
  routers/
    health.py          /healthz, /readyz (pinguea LiteLLM)
    chat.py            /v1/chat/completions (SSE)
    embeddings.py      /v1/embeddings
    models.py          /v1/models (listados por LiteLLM, aliases del config)
    rag.py             /v1/rag/documents (GET/POST/DELETE), /v1/rag/query, /v1/rag/reindex
    agents.py          /v1/agents/tools, /v1/agents/run (SSE de eventos del agente)
  rag/
    parsers.py         txt/md/pdf/html → texto plano (pymupdf, beautifulsoup) +
                       clean_text() (NFC, quita control chars, colapsa espacios)
    chunker.py         Recursive character splitter, sin deps externas
    embedder.py        Wrapper sobre LLMClient.embed con batching + prefijos de
                       tarea nomic (search_query:/search_document:, ver gotcha #19)
    bm25.py            Índice BM25 Okapi en memoria (rama léxica del híbrido)
    store.py           QdrantStore async: ensure_collection, upsert_chunks,
                       delete_by_*, search, scroll_all · contador mutations
    indexer.py         Orquesta parse→chunk→embed→upsert · idempotencia por sha256(chunk)
    retriever.py       query → denso (Qdrant) + BM25 → fusión RRF → mensaje system
                       (híbrido, ver gotcha #21)
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
gpu-exporter/
  Dockerfile           Imagen CUDA base + python con nvidia-smi disponible
  exporter.py          Loop que vuelca métricas de nvidia-smi como gauges Prometheus
prometheus/
  prometheus.yml       Scrape config: api, litellm, gpu-exporter, prometheus self
grafana/
  provisioning/        Datasource (Prometheus) y provider de dashboards
  dashboards/
    llmops-overview.json  Dashboard con paneles HTTP, LLM, agente, RAG, GPU, LiteLLM
docker-compose.yml     9 servicios: ollama, qdrant, litellm, langfuse-db, langfuse,
                       prometheus, grafana, gpu-exporter, api
litellm-config.yaml    Aliases de modelos del proxy (ollama_chat/, openai/,
                       anthropic/) + callbacks langfuse y prometheus
.env.example
Makefile               up/down/logs/*-logs/rebuild/pull-models/bench/smoke/urls
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

11. **Langfuse v2 INIT vars** semillan org/proyecto/keys/usuario en el primer arranque. Eso hace reproducible el setup: tras `make clean` y vuelta a `make up`, las mismas `LANGFUSE_PUBLIC_KEY`/`SECRET_KEY` del `.env` siguen siendo válidas — no hay que entrar a la UI a copiar nada. Si necesitas regenerar keys, hazlo en Settings y vuelca al `.env`.

12. **El trace de Langfuse vive durante todo el SSE stream**, no sólo durante el handler. Por eso `chat_stream` y `agents/run` abren el trace con `start_trace()` (no con context manager) y lo cierran en el `finally` del generador async. Si usas un `with tracer.trace(...)` en el handler, el trace se cierra antes de que se consuma el stream y verás trazas vacías sin output.

13. **LiteLLM ↔ Langfuse cuelga generations por `metadata.existing_trace_id`** (NO `trace_id`), más `parent_observation_id` y `generation_name`. El `LLMClient` mete el dict en `extra_body={"metadata": {...}}` (lo produce `litellm_metadata()` en `observability/langfuse_client.py`). El proxy ancla la generation bajo ese trace si los callbacks están activos en `litellm-config.yaml`. **Por qué `existing_trace_id` y no `trace_id`:** ver gotcha #18.

14. **El gpu-exporter requiere acceso al runtime de NVIDIA igual que Ollama**. Si compose se queja con `could not select device driver "" with capabilities: [[gpu]]`, te falta el NVIDIA Container Toolkit en el host. En Docker Desktop Windows va incluido si tienes WSL2 + driver NVIDIA actualizado.

15. **Langfuse v2 valida el email INIT con Zod** y exige TLD: `admin@local` casca el contenedor (`Invalid environment variables: { LANGFUSE_INIT_USER_EMAIL: [ 'Invalid email' ] }`). Usar `admin@example.com` o cualquier dominio real.

16. **`depends_on` con `condition: service_healthy` te bloquea la API si una dep secundaria casca**. Para Langfuse en particular usamos `service_started` porque el tracer tiene fallback Noop — si Langfuse no levanta, la API debe seguir funcionando sin trazas. Aplica el mismo criterio antes de añadir nuevos servicios al depends_on.

17. **`/metrics` de LiteLLM:** (a) pide auth por defecto — apagarla con `litellm_settings.require_auth_for_metrics_endpoint: false`; (b) redirige `/metrics` → `/metrics/` con host absoluto, lo que Prometheus no resuelve dentro de la red docker. En `prometheus.yml`, `metrics_path: /metrics/` (con barra) evita el redirect.

18. **Race condition de nombres de trace en Langfuse (`trace_id` vs `existing_trace_id`).** Síntoma: en Langfuse aparecen trazas raíz `litellm-acompletion` / `litellm-aembedding` mezcladas con las tuyas (`agent.run`, `chat.completions`), a veces para la *misma* petición. Causa: sobre un mismo `trace_id` escriben **dos** clientes Langfuse — el SDK de la app (que lo crea con su `name`) y el callback del proxy. En `langfuse.py` del proxy, si recibe `trace_id` **sin** `existing_trace_id` ni `trace_name`, hace upsert del trace poniendo `name = f"litellm-{call_type}"`; como ambos flushean async, el último en llegar pisa el `name`. **Fix:** pasar `existing_trace_id` (no `trace_id`) en el metadata → el proxy hace `trace_params={"id": ...}` sin tocar el `name`. Verificado vía la API pública de Langfuse (`GET /api/public/traces`): con `trace_id` ~13 de 16 trazas de agente salían renombradas; con `existing_trace_id` el nombre se mantiene estable. Aparte, **`LLMClient.embed()` debe adjuntar metadata igual que los métodos de chat** (antes no lo hacía) o todo embedding sale como traza huérfana aunque haya trace activo (p.ej. el embed de `rag_search` dentro de un `agent.run`).

19. **`nomic-embed-text` exige prefijos de tarea.** El modelo se entrenó con una instrucción incrustada en el texto: documentos → `search_document: <texto>`, queries → `search_query: <texto>`. **Ollama NO los inyecta**, hay que ponerlos en el cliente (`embedder.py`, `_prepare()`). Sin ellos la recuperación se degrada de forma sutil pero grave: scores apelotonados (todo ~0.5-0.6) y vecinos semánticamente erróneos. Query y documento DEBEN usar su prefijo correspondiente — mezclar (o re-indexar solo un lado) rompe la comparación. Verificación rápida: `cos(vector_guardado, embed("search_document: "+texto))` debe dar ~1.0. Cambiar esto obliga a **re-indexar todo** (los vectores viejos están en otro espacio).

20. **Editar `litellm-config.yaml` NO basta con `docker compose up -d litellm`.** La config entra por bind-mount; al cambiar su *contenido* la definición del servicio no cambia, así que compose deja el contenedor "up-to-date" sin recrearlo, y LiteLLM ya parseó el YAML viejo en memoria al arrancar. Síntoma típico: cambias el model id de un alias (p.ej. prefijo `us.`/`global.` de Bedrock) y el error sigue mostrando el id antiguo. **Fix:** `docker compose restart litellm` (o `--force-recreate`). Verifica lo que cargó de verdad con `curl http://localhost:4000/v1/model/info -H "Authorization: Bearer $LITELLM_MASTER_KEY"`. (Relacionado: para Bedrock con inference profiles, pasa el id con prefijo regional `us.`/`eu.`/`apac.` o el ARN completo; el prefijo `global.` puede no reconocerlo según versión y acaba mandando el id pelado.)

21. **RAG es híbrido (denso + BM25 con RRF), no solo vectorial.** `retriever.py` consulta la rama densa en Qdrant Y un índice BM25 en memoria (`bm25.py`), y fusiona ambos rankings con Reciprocal Rank Fusion ponderada (`rag_*` en `config.py`). Por qué: nomic difumina queries con términos exactos o "meta" ausentes del texto (nombres, siglas, "currículum"); BM25 los rescata. RRF fusiona por *posición*, no por magnitud, porque coseno y BM25 no son comparables en escala. El índice BM25 se reconstruye cuando `store.mutations` cambia (no en cada query). Consecuencia visible: el `score` de los hits ahora es el valor RRF (~0.04), no el coseno. Si recuperación va mal, distingue las dos ramas por separado antes de tocar la fusión (lo hice query a query contra `/v1/rag/query` y scroll de Qdrant).

22. **Los uploads de la UI (`upload://…`) no se persisten en disco.** Van directos a Qdrant; los bytes originales no se guardan. Por tanto `POST /v1/rag/reindex` —que recorre `RAG_DOCS_DIR`— **no los alcanza**: solo reindexa lo que esté en `./docs`. Para re-extraer/re-indexar un documento subido por la UI (p.ej. tras cambiar el parser) hay que volver a subirlo, o dejarlo en `./docs` (el watcher lo coge, pero con `source` distinto: `/app/docs/x.pdf` ≠ `upload://x.pdf`, así que conviene borrar el viejo para no duplicar). La extracción de PDF usa **PyMuPDF** (`parsers.py`), mejor que pypdf en espaciado/encoding; aun así PDFs con fuentes raras (CVs con plantilla) pueden dejar algún artefacto menor que `clean_text()` no resuelve del todo.

23. **Optimizaciones de latencia (jul 2026) y sus supuestos.** (a) *Warm-up*: al arrancar la API lanza en background 1 chat de 1 token + 1 embed a los modelos por defecto (`_warmup` en `main.py`, `WARMUP_ENABLED`); sin trace activo, LiteLLM registra una traza propia al arrancar — es esperado. (b) *Tools paralelas*: cuando el modelo emite varias tool_calls en una respuesta, `AgentLoop` las ejecuta con `asyncio.gather`; el orden de eventos SSE cambió de `call₁,result₁,call₂,result₂` a `call₁,call₂,result₁,result₂` (la UI ya keyea cards por id, no le afecta; los `duration_ms` ahora se solapan). (c) *Embedder*: LRU de embeddings solo para `task="query"` — es correcto sin invalidación porque el embedding de una query no depende del corpus; documentos no se cachean. (d) *Retriever híbrido*: rama densa (embed→Qdrant) y léxica (rebuild BM25) corren con `gather`. (e) `/v1/models` se cachea 30 s (module-level en `routers/models.py`): tras editar `litellm-config.yaml` + restart del proxy, el listado tarda ≤30 s en reflejarse. (f) `is_disconnected()` se comprueba como mucho cada 250 ms en los streams SSE (hace un receive() ASGI por llamada); la cancelación sigue siendo ~inmediata.

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
5. `api/app/observability/langfuse_client.py` — modelo de tracing (trace→span→generation)
6. `api/app/observability/metrics.py` — catálogo completo de métricas Prometheus
7. `grafana/dashboards/llmops-overview.json` — qué se ve en el dashboard
8. `api/app/static/index.html` — UI completa, sirve también de "manual de uso" del API
9. `docker-compose.yml` — orquestación y healthchecks
