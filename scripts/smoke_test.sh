#!/usr/bin/env bash
# Smoke test contra la API: chat (no-stream y stream), embeddings, models.
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
MODEL="${MODEL:-llama3.1:8b-instruct-q4_K_M}"
EMBED_MODEL="${EMBED_MODEL:-nomic-embed-text}"

echo "== /healthz"
curl -sf "$BASE_URL/healthz" | jq .

echo "== /readyz"
curl -sf "$BASE_URL/readyz" | jq .

echo "== /v1/models"
curl -sf "$BASE_URL/v1/models" | jq '.data | length as $n | "\($n) modelos"'

echo "== /v1/chat/completions (no stream)"
curl -sf "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$(jq -n --arg m "$MODEL" '{
    model: $m,
    messages: [{role:"user", content:"Di hola en una frase corta."}],
    max_tokens: 40,
    temperature: 0.2
  }')" | jq '{model, content: .choices[0].message.content, usage}'

echo "== /v1/chat/completions (stream, SSE crudo)"
curl -sN "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$(jq -n --arg m "$MODEL" '{
    model: $m,
    messages: [{role:"user", content:"Cuenta del 1 al 5, separados por espacios."}],
    max_tokens: 30,
    stream: true
  }')" | head -n 30

echo
echo "== /v1/embeddings"
curl -sf "$BASE_URL/v1/embeddings" \
  -H 'Content-Type: application/json' \
  -d "$(jq -n --arg m "$EMBED_MODEL" '{
    model: $m,
    input: ["hola mundo", "infraestructura de IA"]
  }')" | jq '{model, n: (.data | length), dim: (.data[0].embedding | length), usage}'

echo "== smoke OK"
