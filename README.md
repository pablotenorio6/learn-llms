# llm_ops

Asistente LLM con harness propio y stack de ops alrededor: wrapper FastAPI OpenAI-compatible, multi-proveedor vía proxy LiteLLM (Ollama local + OpenAI + Anthropic), RAG sobre Qdrant, agente con tools y observabilidad completa (Langfuse + Prometheus + Grafana).

Proyecto de aprendizaje de despliegue/infra de IA. Plan completo en [`roadmap.md`](./roadmap.md); contexto técnico detallado en [`CLAUDE.md`](./CLAUDE.md).

**Estado:** Fases 0–4 completas (wrapper API, observabilidad, RAG, tool calling + agente) sobre arquitectura multi-proveedor con LiteLLM. Pendientes: evals + CI (Fase 5), producción-grade (Fase 6).

## Arquitectura

```
[Browser] ──▶ [FastAPI :8000] ──▶ [LiteLLM :4000] ──┬──▶ Ollama :11434 (local, GPU)
                    │                                ├──▶ api.openai.com
                    │                                └──▶ api.anthropic.com
                    ├──▶ Qdrant :6333   (RAG)
                    └──▶ Langfuse :3030 (trazas)

Observabilidad: Prometheus :9090 raspa /metrics de api + litellm + gpu-exporter;
Grafana :3001 sirve el dashboard provisionado.
```

La API habla **OpenAI puro** contra LiteLLM; el proxy enruta al backend según el alias del modelo. Añadir un proveedor es una entrada en [`litellm-config.yaml`](./litellm-config.yaml) — la API no necesita rebuild.

## Quickstart

Requisitos: Docker Desktop con soporte NVIDIA (WSL2 + drivers), GPU NVIDIA, ~12 GB libres para los modelos locales.

```bash
cp .env.example .env
# Opcional: rellena OPENAI_API_KEY / ANTHROPIC_API_KEY para usar modelos cloud,
# y BRAVE_API_KEY para la tool web_search. En local, los defaults valen.
docker compose up -d --build
make pull-models           # solo la primera vez (~12 GB)
make smoke                 # sanity check (opcional)
```

UI en `http://localhost:8000/`. La API OpenAI-compatible está en `http://localhost:8000/v1`:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
r = client.chat.completions.create(
    model="qwen-local",   # alias del proxy; ver tabla de modelos
    messages=[{"role": "user", "content": "Hola"}],
)
print(r.choices[0].message.content)
```

## Modelos

Los aliases viven en `litellm-config.yaml`; la API y la UI siempre hablan el alias, nunca el id del proveedor.

| Alias | Backend | Notas |
|---|---|---|
| `qwen-local` | Ollama · qwen2.5:7b-instruct | Recomendado para modo agente (mejor disciplina de tools) |
| `llama-local` | Ollama · llama3.1:8b-instruct | Buen tool calling, tiende a over-calling |
| `phi-local` | Ollama · phi3.5:3.8b | Sin tool calling nativo |
| `nomic-embed` | Ollama · nomic-embed-text | Embeddings 768-dim (RAG) |
| `gpt-4o-mini`, `gpt-4o` | OpenAI | Requieren `OPENAI_API_KEY` |
| `claude-sonnet-4-6`, `claude-haiku-4-5` | Anthropic | Requieren `ANTHROPIC_API_KEY` |

`make pull-models` descarga los locales.

## Endpoints

```
GET  /healthz, /readyz                Salud / readiness (pinguea LiteLLM)
POST /v1/chat/completions             Chat OpenAI-compatible (SSE streaming)
POST /v1/embeddings                   Embeddings
GET  /v1/models                       Aliases declarados en el proxy
GET/POST/DELETE /v1/rag/documents     CRUD de documentos en la KB
POST /v1/rag/query                    Retrieval sobre Qdrant
POST /v1/rag/reindex                  Reindexa RAG_DOCS_DIR
GET  /v1/agents/tools                 Tools registradas
POST /v1/agents/run                   Bucle del agente (SSE de eventos)
GET  /metrics                         Métricas Prometheus
```

## URLs del stack

`make urls` las imprime todas:

| Servicio | URL | Credenciales |
|---|---|---|
| API + UI | http://localhost:8000/ | — |
| Qdrant dashboard | http://localhost:6333/dashboard | — |
| LiteLLM | http://localhost:4000/health/liveliness | master key (`.env`) |
| Langfuse | http://localhost:3030/ | `admin@example.com` / ver `.env` |
| Prometheus | http://localhost:9090/ | — |
| Grafana | http://localhost:3001/ | `admin` / ver `.env` |
| GPU exporter | http://localhost:9835/metrics | — |

## Estructura

```
api/                  FastAPI wrapper OpenAI-compatible
  app/
    main.py           Entry point · lifespan (LLMClient + RAG + Watcher + Langfuse)
    config.py         Settings via pydantic-settings (.env)
    models.py         Schemas Pydantic (OpenAI + RAG + Agents)
    middleware/       Logging structlog JSON, X-Request-ID
    observability/    Tracer Langfuse + métricas Prometheus
    services/         LLMClient async contra el proxy LiteLLM
    routers/          chat, embeddings, models, rag, agents, health
    rag/              parsers · chunker · embedder · store (Qdrant) · indexer · retriever · watcher
    agents/           registry (@tool) · loop (AgentLoop SSE) · tools (rag_search, web_search, http_fetch)
    static/           UI vanilla (chat + RAG + agente + SSE)
    tests/            E2E sobre TestClient + FakeLLM
bench/                Benchmark TTFT / tokens-s vía el proxy
docs/                 Carpeta vigilada por el watcher (RAG) — bind-mount
gpu-exporter/         Side-car nvidia-smi → métricas Prometheus
prometheus/           Scrape config
grafana/              Datasource + dashboard provisionados
docker-compose.yml    9 servicios
litellm-config.yaml   Aliases de modelos + callbacks Langfuse/Prometheus
Makefile              Atajos
```

## Comandos útiles

```bash
make up / down        # levanta / para el stack completo
make logs             # logs en vivo de todos los servicios
make litellm-logs     # (o langfuse-/prom-/grafana-/gpu-logs) logs por servicio
make rebuild          # rebuild + restart SOLO de la api (tras tocar api/app/)
make pull-models      # descarga los modelos locales en Ollama
make bench            # benchmark sobre BENCH_MODELS
make smoke            # smoke test de todos los endpoints
make shell            # shell dentro del contenedor api
make urls             # imprime las URLs del stack
make ps               # estado de los contenedores
make clean            # down + borra volúmenes (¡pierde modelos e índices!)
```
