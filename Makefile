.PHONY: help up down logs rebuild bench smoke shell pull-models ps clean litellm-logs

help:
	@echo "Targets:"
	@echo "  up            - levanta el stack (Ollama + Qdrant + LiteLLM + API)"
	@echo "  down          - para el stack"
	@echo "  logs          - logs en vivo de todos los servicios"
	@echo "  litellm-logs  - logs solo del proxy LiteLLM"
	@echo "  rebuild       - rebuild de la API y restart"
	@echo "  pull-models   - descarga los modelos baseline en Ollama"
	@echo "  bench         - corre el benchmark sobre BENCH_MODELS"
	@echo "  smoke         - smoke test (chat + embeddings + streaming)"
	@echo "  shell         - shell dentro del contenedor api"
	@echo "  ps            - estado de los contenedores"
	@echo "  clean         - down + borra volumen de Ollama (¡pierde modelos!)"

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

litellm-logs:
	docker compose logs -f --tail=100 litellm

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

ps:
	docker compose ps

clean:
	docker compose down -v
