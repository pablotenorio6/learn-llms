# llm_ops

Asistente LLM local con harness propio + stack de ops alrededor.
Roadmap completo en [`roadmap.md`](./roadmap.md).

Estado actual: **Fase 0 + Fase 1 (wrapper API OpenAI-compatible)**.

## Quickstart

Requisitos: Docker con soporte NVIDIA (`nvidia-container-toolkit`), GPU NVIDIA con drivers, ~10 GB libres.

```bash
cp .env.example .env
make up           # arranca Ollama + API
make pull-models  # descarga los modelos baseline
make smoke        # smoke test contra la API
```

API disponible en `http://localhost:8000/v1` (compatible con el SDK de OpenAI).

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
r = client.chat.completions.create(
    model="llama3.1:8b-instruct-q4_K_M",
    messages=[{"role": "user", "content": "Hola"}],
)
print(r.choices[0].message.content)
```

## Estructura

```
api/                FastAPI wrapper (OpenAI-compatible)
  app/
    main.py         App entry
    config.py       Settings via env
    models.py       Pydantic schemas (OpenAI dialect)
    services/       Cliente Ollama
    routers/        Endpoints: chat, embeddings, models, health
    middleware/     Logging estructurado, request IDs
bench/              Scripts de benchmarking del modelo
scripts/            Helpers (pull modelos, smoke test)
prompts/            Prompts versionados (Fase 5)
docker-compose.yml  Orquestación local
Makefile            Atajos comunes
```

## Comandos útiles

```bash
make up         # docker compose up -d
make down       # docker compose down
make logs       # logs en vivo
make rebuild    # rebuild de la api
make bench      # corre el benchmark sobre los modelos del .env
make smoke      # smoke test (chat + embeddings + streaming)
make shell      # shell dentro del contenedor api
```
