#!/usr/bin/env bash
# Descarga los modelos baseline en Ollama (vía el contenedor del compose).
set -euo pipefail

models=(
  "llama3.1:8b-instruct-q4_K_M"
  "qwen2.5:7b-instruct-q4_K_M"
  "phi3.5:3.8b"
  "nomic-embed-text"
)

for m in "${models[@]}"; do
  echo "==> pulling $m"
  docker compose exec -T ollama ollama pull "$m"
done

echo "==> done"
docker compose exec -T ollama ollama list
