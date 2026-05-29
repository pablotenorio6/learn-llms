.PHONY: help up down logs rebuild bench smoke shell pull-models ps clean \
        litellm-logs langfuse-logs prom-logs grafana-logs gpu-logs \
        urls

help:
	@echo "Targets:"
	@echo "  up             - levanta el stack completo"
	@echo "  down           - para el stack"
	@echo "  logs           - logs en vivo de todos los servicios"
	@echo "  litellm-logs   - logs sólo del proxy LiteLLM"
	@echo "  langfuse-logs  - logs sólo de Langfuse + su Postgres"
	@echo "  prom-logs      - logs sólo de Prometheus"
	@echo "  grafana-logs   - logs sólo de Grafana"
	@echo "  gpu-logs       - logs del gpu-exporter"
	@echo "  rebuild        - rebuild de la API y restart"
	@echo "  pull-models    - descarga los modelos baseline en Ollama"
	@echo "  bench          - corre el benchmark sobre BENCH_MODELS"
	@echo "  smoke          - smoke test (incl. métricas + Langfuse + GPU)"
	@echo "  shell          - shell dentro del contenedor api"
	@echo "  urls           - imprime las URLs útiles del stack"
	@echo "  ps             - estado de los contenedores"
	@echo "  clean          - down + borra todos los volúmenes (¡pierde modelos!)"

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

litellm-logs:
	docker compose logs -f --tail=100 litellm

langfuse-logs:
	docker compose logs -f --tail=100 langfuse langfuse-db

prom-logs:
	docker compose logs -f --tail=100 prometheus

grafana-logs:
	docker compose logs -f --tail=100 grafana

gpu-logs:
	docker compose logs -f --tail=100 gpu-exporter

rebuild:
	docker compose up -d --build api

pull-models:
	bash scripts/pull_models.sh

bench:
	docker compose exec api python -m bench.benchmark

smoke:
	bash scripts/smoke_test.sh

shell:
	docker compose exec api /bin/bash

urls:
	@echo "API           http://localhost:8000/"
	@echo "API metrics   http://localhost:8000/metrics"
	@echo "Qdrant UI     http://localhost:6333/dashboard"
	@echo "LiteLLM       http://localhost:4000/health/liveliness"
	@echo "Langfuse      http://localhost:3030/  (admin@local / ver .env)"
	@echo "Prometheus    http://localhost:9090/"
	@echo "Grafana       http://localhost:3001/  (admin / ver .env)"
	@echo "GPU exporter  http://localhost:9835/metrics"

ps:
	docker compose ps

clean:
	docker compose down -v
