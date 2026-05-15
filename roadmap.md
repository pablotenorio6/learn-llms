# Proyecto: LLM local + harness + ops

Roadmap por fases para aprender despliegue e infraestructura de IA construyendo un asistente local realmente usable.

**Hardware objetivo**: GTX 4060 8 GB VRAM, 32 GB RAM DDR5.

**Filosofía**: cada fase entrega algo funcional end-to-end. Nada de "primero monto toda la infra y luego veo si funciona". Cada semana cierras con algo demostrable.

---

## Fase 0 — Baseline y benchmarking (1–2 días)

**Objetivo**: entender qué da tu hardware antes de elegir nada.

Tareas:
- Instalar Ollama. `ollama pull llama3.1:8b-instruct-q4_K_M` y `ollama pull qwen2.5:7b-instruct-q4_K_M`.
- También: `phi3.5:3.8b` (rápido) y `qwen2.5:14b-instruct-q4_K_M` (forzará offload parcial CPU/GPU).
- Medir con un script propio: tokens/s en prompt processing y en generation, latencia al primer token, VRAM/RAM ocupada (`nvidia-smi`, `htop`), throughput con concurrencia 1, 2, 4.
- Probar también `nomic-embed-text` (137M) para embeddings — vas a necesitarlo en Fase 3.

Criterio de éxito: tienes una hoja con tokens/s y consumo de RAM/VRAM por modelo y por nivel de cuantización. Eso te guía toda la decisión posterior.

Deliverable: `bench/results.md` con números reales, no estimados.

---

## Fase 1 — Wrapper API compatible OpenAI (1 semana)

**Objetivo**: tu propio servidor delante del modelo, hablando el dialecto que habla todo el mundo.

Stack:
- FastAPI + uvicorn.
- Ollama por debajo (puerto 11434) o `llama-server` de llama.cpp si quieres más control.
- Pydantic v2 para los esquemas (copia los de OpenAI: `ChatCompletionRequest`, `ChatCompletionChunk`, etc.).

Endpoints mínimos:
- `POST /v1/chat/completions` con y sin `stream=true` (SSE).
- `POST /v1/embeddings`.
- `GET /v1/models`.
- `GET /healthz` y `GET /readyz`.

Detalles que enseñan mucho:
- Streaming real con `StreamingResponse` y `text/event-stream`, respetando el formato `data: {...}\n\n` y el `[DONE]` final.
- Cancelación: si el cliente cierra la conexión, cortas la generación río abajo.
- Manejo de errores con el mismo schema de errores que OpenAI (te ahorra problemas con clientes existentes).
- Timeouts y límites de tokens configurables.

Dockerización:
- `Dockerfile` multi-stage para tu API.
- `docker-compose.yml` con: tu API + Ollama + un Caddy/Traefik delante haciendo TLS local.
- Variables por `.env`, nada hardcoded.

Test de la victoria: apuntar el SDK de OpenAI a `http://localhost:8000/v1` con una API key dummy y que funcione sin tocar el código del cliente.

---

## Mejora multi-proveedor — LiteLLM proxy (extensión de Fase 1)

**Objetivo:** dejar de ser solo-Ollama. Mismo cliente, varios backends (Ollama local, OpenAI, Anthropic, etc.), con un punto único de control.

Cambios:
- Nuevo contenedor `litellm` (imagen `ghcr.io/berriai/litellm`) con `litellm-config.yaml` declarando aliases (`llama-local`, `qwen-local`, `gpt-4o-mini`, `claude-sonnet-4-6`, …) mapeados a `ollama_chat/...`, `openai/...`, `anthropic/...`.
- La API ya no habla con Ollama. Habla con `http://litellm:4000/v1` usando el SDK oficial `openai` async, autenticada con `LITELLM_MASTER_KEY`.
- `OllamaClient` → `LLMClient`. Métodos limpios (no monkey-patches): `chat`, `chat_stream`, `chat_with_tools(_stream)`, `embed`, `list_models`.
- `AgentLoop` actualizado para acumular `tool_calls` progresivos en streaming OpenAI (deltas por `index` con `function.arguments` concatenándose). Soporta tools paralelas en una sola respuesta.
- `/v1/models` queda alimentado por LiteLLM: la fuente de verdad de qué modelos existen es el YAML del proxy.

Beneficios secundarios:
- Cost tracking nativo de LiteLLM (lo conectaremos a Langfuse en Fase 2).
- Retries y fallbacks declarativos por modelo en el YAML.
- Las API keys de OpenAI/Anthropic viven en un solo sitio (el proxy), no esparcidas por la app.
- Cambiar de proveedor o añadir uno nuevo es un edit del YAML + restart del contenedor; la API no se rebuildea.

Test de la victoria: cambiar el modelo en la UI entre `llama-local` y `gpt-4o-mini` sin tocar código, y que tool calling funcione idéntico en ambos.

---

## Fase 2 — Observabilidad (1 semana)

**Objetivo**: si no lo mides, no lo entiendes. Esta es la fase que más distingue un proyecto de juguete de uno serio.

Stack recomendado:
- **Langfuse** self-hosted (docker-compose oficial) para trazas LLM-específicas: prompts, completions, latencia, coste estimado, evals.
- **Prometheus + Grafana** para métricas de sistema: latencia p50/p95/p99, RPS, errores, VRAM, queue depth.
- **Logs estructurados** en JSON con `structlog`, correlation ID por request.

Instrumentación a añadir en tu API:
- Middleware que abre un trace de Langfuse por request.
- Métricas Prometheus: `llm_request_duration_seconds` (histogram), `llm_tokens_generated_total` (counter), `llm_active_requests` (gauge).
- Timing fino: tiempo a primer token (TTFT) vs tiempo total, separado.

Dashboards Grafana mínimos:
- Throughput y latencia por endpoint.
- Tokens/s reales (no teóricos) en producción.
- Saturación de GPU.

Criterio de éxito: puedes responder "¿por qué la última petición tardó 4 segundos?" mirando una traza.

---

## Fase 3 — RAG sobre tus propias notas (1–2 semanas)

**Objetivo**: el asistente deja de inventarse cosas sobre tu mundo.

Stack:
- **Qdrant** local (docker) como vector store. Más maduro que Chroma y con buen filtrado por metadata.
- **Embeddings**: `nomic-embed-text` vía Ollama, o `bge-small-en-v1.5` si vas en inglés.
- **Parser de documentos**: `unstructured` o `docling` para PDFs, markdown, html.

Pipeline de ingesta:
1. Watcher de una carpeta (notas, PDFs, lo que sea).
2. Parsing → chunking (semántico, no por caracteres ciegamente — prueba `semantic-text-splitter`).
3. Embed → upsert en Qdrant con metadata (source, fecha, tipo).
4. Idempotencia: hash del chunk para no re-indexar lo mismo.

Pipeline de retrieval:
- Búsqueda híbrida: vector + BM25 (Qdrant lo soporta nativo desde 1.10).
- Reranking con un cross-encoder pequeño (`bge-reranker-base`).
- Inyección en prompt con citas verificables.

Cosas que vas a aprender por las malas (y son el oro de esta fase):
- Por qué el chunking importa más que el modelo.
- Por qué el reranking sube la calidad muchísimo.
- Cómo evaluar un RAG (Fase 5 te servirá aquí).

Endpoint nuevo: `POST /v1/chat/completions` con un parámetro `tools` que active retrieval, o un endpoint `/v1/rag/query` separado.

---

## Fase 4 — Tool calling y harness de agente (1–2 semanas)

**Objetivo**: el modelo deja de ser un chatbot y empieza a hacer cosas.

Implementación:
- Soportar el schema de `tools` de OpenAI en tu API (ya tienes la base de Fase 1).
- Para modelos que no saben function-calling nativo, implementar un parser de salida estructurada con prompts (formato XML o JSON con validación Pydantic + reintentos).
- Modelos que sí saben hacerlo nativo en local: Llama 3.1 8B Instruct y Qwen 2.5 7B Instruct van bien.

Tool registry propio:
- Decorador `@tool` que registra funciones Python con su schema autogenerado desde type hints.
- Sandboxing básico: las tools no llaman cosas peligrosas sin `dry_run`.
- Tools de ejemplo: search en tu RAG, leer/escribir archivos en una sandbox, ejecutar Python (en contenedor aparte), web search vía SearXNG local.

Loop de agente:
- Implementar el bucle clásico: model → tool calls → ejecutar → resultados → model → ...
- Límite de iteraciones, presupuesto de tokens por sesión.
- Logging de cada paso en Langfuse como spans anidados (queda precioso).

Guardrails básicos:
- Validación de output con Pydantic.
- Filtrado de prompts maliciosos (regex + clasificador pequeño tipo `ProtectAI`).
- Allow-list de tools por endpoint.

---

## Fase 5 — Evals y CI (1 semana)

**Objetivo**: dejar de cambiar prompts a ojo.

Stack:
- **Promptfoo** o **DeepEval** para el framework. Promptfoo es más simple y se integra mejor con CI.
- Datasets propios: 50–100 ejemplos cubriendo casos representativos (RAG hits, RAG misses, tool calling, edge cases).

Tipos de evals:
- Determinísticas: contiene esta cita, llamó a esta tool con estos args, no menciona X.
- LLM-as-judge: usa un modelo más grande (Claude o GPT-4 vía API) como juez para criterios subjetivos. Ojo al coste — limítalo a regresiones.
- Métricas RAG: faithfulness, answer relevancy, context precision/recall (ragas las tiene listas).

Integración en CI:
- GitHub Actions o gitea-act local.
- En cada PR: corre evals → reporta diff vs main → bloquea merge si caída > X%.
- Una versión "barata" que corre en cada commit y otra "completa" semanal.

Versionado:
- Prompts en archivos versionados, no hardcoded en código.
- Cada eval run guarda modelo + prompt hash + resultados → puedes auditar regresiones.

---

## Fase 6 — Producción-grade (1–2 semanas)

**Objetivo**: que el sistema aguante uso real (aunque solo seas tú).

Piezas:
- **Caché** de respuestas con Redis. Cache key = hash(model + messages + temperature). TTL configurable. Bypass con header.
- **Rate limiting** por API key con `slowapi` o nginx.
- **Auth** simple: API keys en una tabla SQLite/Postgres, scopes por key.
- **Router de modelos**: una capa que decide entre tu Llama local y, por ejemplo, Claude Haiku vía API según reglas (longitud de prompt, latencia requerida, tipo de tarea). Útil para entender el patrón "fallback to cloud".
- **Cola** opcional: si quieres simular carga real, mete RabbitMQ o Redis Streams entre la API y el worker que llama al modelo.
- **Load testing** con `k6` o `locust`. Curvas de RPS vs latencia, encontrar el knee point de tu 4060.

Deploy:
- Si tienes una segunda máquina o un mini-PC: monta k3s y despliega ahí con Helm. Si no, queda con docker-compose pero estructurado como si fuera para k8s (separación de servicios, healthchecks, configmaps simulados con `.env` files).

---

## Fase 7 (opcional) — Frontend (1 semana)

**Objetivo**: una UI decente para no vivir en `curl`.

Opciones:
- **Open WebUI**: lo apuntas a tu API OpenAI-compatible y lo tienes gratis con sesiones, RAG UI, modelos múltiples. Recomendado si quieres usarlo de verdad.
- **Construirla tú**: Next.js + Vercel AI SDK. Aprendes streaming en el cliente, que es divertido.

---

## Stack final cristalizado

```
[ Open WebUI ]  ──HTTP──▶  [ FastAPI wrapper ]  ──▶  [ Ollama / llama.cpp ]
                                  │
                                  ├──▶ [ Qdrant ] (vectores)
                                  ├──▶ [ Redis ] (caché + rate limit)
                                  ├──▶ [ Postgres ] (auth, sesiones, eval runs)
                                  ├──▶ [ Langfuse ] (trazas LLM)
                                  └──▶ [ Prometheus + Grafana ] (métricas sistema)

[ GitHub Actions ] ──▶ [ Promptfoo ] ──▶ [ Langfuse ] (evals)
```

Todo en un solo `docker-compose.yml` que arranca con `docker compose up`.

---

## Modelos recomendados para tu 4060

| Modelo | Cuantización | VRAM | Uso |
|---|---|---|---|
| Llama 3.1 8B Instruct | Q4_K_M | ~4.9 GB | Generalista, buen tool calling |
| Qwen 2.5 7B Instruct | Q4_K_M | ~4.7 GB | Mejor en código y razonamiento |
| Phi-3.5-mini 3.8B | Q4_K_M | ~2.4 GB | Rápido, para tareas simples / clasificación |
| Qwen 2.5 14B Instruct | Q4_K_M | ~9 GB | Offload parcial, más calidad, más lento |
| nomic-embed-text | F16 | ~280 MB | Embeddings RAG |
| bge-reranker-base | F16 | ~1.1 GB | Reranking |

Puedes mantener Llama 3.1 8B + nomic-embed cargados a la vez sin problema. Para el reranker, cárgalo bajo demanda o úsalo en CPU.

---

## Cadencia sugerida

Si dedicas ~10h/semana, son **8–10 semanas** para llegar al final. Si solo 3–5h/semana, dóblalo. Lo importante: **cierra cada fase antes de empezar la siguiente**, aunque sea con la versión mínima. Los proyectos de infra mueren cuando se intenta perfeccionar todo a la vez.

---

## Criterio de éxito global

Al final del proyecto deberías poder responder con confianza, mirando código tuyo:
- ¿Qué pasa exactamente cuando llega un request a tu API?
- ¿Cómo se distribuye la latencia entre red, API, modelo, tools?
- ¿Cuántos tokens/s aguanta tu hardware en concurrencia 4?
- ¿Cuánto degrada la calidad si bajas a Q3 vs Q4 vs Q5?
- ¿Qué cambió en evals tu último cambio de prompt?
- ¿Por qué eligiste vector + BM25 en vez de solo vector?

Si puedes responder eso, este proyecto te ha enseñado más sobre infra de IA que el 95% de los cursos online.
