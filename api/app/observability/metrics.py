"""Métricas Prometheus LLM-específicas.

Las HTTP-métricas estándar (req/s, latencia por endpoint, status) las añade
`prometheus-fastapi-instrumentator` automáticamente. Aquí definimos métricas
de dominio que no se derivan del HTTP layer:

  - llm_request_duration_seconds  — duración total de una llamada al modelo
  - llm_ttft_seconds              — tiempo al primer token (sólo streaming)
  - llm_tokens_total              — tokens consumidos/generados, por dirección
  - llm_active_requests           — gauge de requests LLM en vuelo
  - agent_iterations_total        — iteraciones del bucle del agente
  - agent_tool_calls_total        — invocaciones a tools, etiquetadas por outcome
  - agent_tool_duration_seconds   — latencia por tool
  - rag_retrieval_duration_seconds
  - rag_chunks_returned

Usamos el registry por defecto (REGISTRY global de prometheus_client) para que
el instrumentator y nuestras métricas convivan en el mismo /metrics.
"""

from __future__ import annotations

from prometheus_client import REGISTRY, Counter, Gauge, Histogram

METRICS_REGISTRY = REGISTRY  # alias semántico

# ---- LLM ----

# Buckets escogidos a mano: TTFT típico en local va de 0.2s a 5s en concurrencia;
# duración total puede ir de 1s a 60s+. Los buckets cubren ambos rangos.
_DUR_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0)
_TTFT_BUCKETS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)

llm_request_duration = Histogram(
    "llm_request_duration_seconds",
    "Duración end-to-end de una llamada al LLM (excluye HTTP overhead).",
    labelnames=("model", "endpoint", "stream", "status"),
    buckets=_DUR_BUCKETS,
)

llm_ttft = Histogram(
    "llm_ttft_seconds",
    "Tiempo al primer token recibido (sólo aplica a streaming).",
    labelnames=("model", "endpoint"),
    buckets=_TTFT_BUCKETS,
)

llm_tokens_total = Counter(
    "llm_tokens_total",
    "Tokens consumidos por el LLM, etiquetados por dirección (input/output).",
    labelnames=("model", "direction"),
)

llm_active_requests = Gauge(
    "llm_active_requests",
    "Requests LLM activos en este momento (incrementa al empezar, decrementa al acabar).",
    labelnames=("endpoint",),
)

# ---- Agent ----

agent_iterations_total = Counter(
    "agent_iterations_total",
    "Iteraciones del bucle del agente (model→tools→model). 1 por vuelta.",
    labelnames=("model",),
)

agent_tool_calls_total = Counter(
    "agent_tool_calls_total",
    "Llamadas a tools del agente.",
    labelnames=("tool", "outcome"),  # outcome: ok | error | bad_args | unknown
)

agent_tool_duration = Histogram(
    "agent_tool_duration_seconds",
    "Duración de la ejecución de una tool.",
    labelnames=("tool",),
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
)

# ---- RAG ----

rag_retrieval_duration = Histogram(
    "rag_retrieval_duration_seconds",
    "Duración total de un retrieval (embed + search en Qdrant).",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)

rag_chunks_returned = Histogram(
    "rag_chunks_returned",
    "Número de chunks devueltos por retrieval.",
    buckets=(0, 1, 2, 3, 5, 10, 20, 50),
)
