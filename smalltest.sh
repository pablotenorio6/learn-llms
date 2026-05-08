PROMPT="${1:-Di algo corto}"

curl -sf "http://localhost:8000/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$(jq -n --arg m "llama3.1:8b-instruct-q4_K_M" \
    --arg p "$PROMPT" \
    '{
    model: $m,
    messages: [{role:"user", content:$p}],
    max_tokens: 200,
    stream: true,
    temperature: 0.2
  }')" | head -n 30
