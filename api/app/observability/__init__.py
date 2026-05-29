"""Observabilidad: Langfuse (trazas LLM) + Prometheus (métricas)."""

from app.observability.langfuse_client import (
    LangfuseTracer,
    current_trace_id,
    current_observation_id,
    get_tracer,
    init_tracer,
    shutdown_tracer,
)
from app.observability.metrics import (
    METRICS_REGISTRY,
    agent_iterations_total,
    agent_tool_calls_total,
    agent_tool_duration,
    llm_active_requests,
    llm_request_duration,
    llm_tokens_total,
    llm_ttft,
    rag_chunks_returned,
    rag_retrieval_duration,
)

__all__ = [
    "LangfuseTracer",
    "current_trace_id",
    "current_observation_id",
    "get_tracer",
    "init_tracer",
    "shutdown_tracer",
    "METRICS_REGISTRY",
    "agent_iterations_total",
    "agent_tool_calls_total",
    "agent_tool_duration",
    "llm_active_requests",
    "llm_request_duration",
    "llm_tokens_total",
    "llm_ttft",
    "rag_chunks_returned",
    "rag_retrieval_duration",
]
