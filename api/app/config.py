"""Configuración de la aplicación, cargada desde variables de entorno."""

from __future__ import annotations

import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # API
    api_port: int = 8000
    api_log_level: str = "INFO"
    api_keys: str = "dev-key-changeme"
    api_require_auth: bool = False

    # LiteLLM proxy (la API habla con LiteLLM en formato OpenAI; LiteLLM enruta
    # a Ollama / OpenAI / Anthropic según el alias del modelo).
    litellm_base_url: str = "http://litellm:4000"
    litellm_master_key: str = "sk-llmops-changeme"
    litellm_request_timeout: int = 600

    # Modelos por defecto (alias declarados en litellm-config.yaml).
    default_chat_model: str = "qwen-local"
    default_embed_model: str = "nomic-embed"
    bench_models: str = Field(default="llama-local,qwen-local,phi-local")

    # Latencia
    # Warm-up al arrancar: fuerza la carga en VRAM del modelo de chat y el de
    # embeddings para que el primer request real no pague el cold-load de
    # Ollama (varios segundos). Corre en background, no bloquea el startup.
    warmup_enabled: bool = True
    # TTL del cache de GET /v1/models (la UI lo pide en cada carga; los aliases
    # solo cambian al editar litellm-config.yaml). 0 = sin cache.
    models_cache_ttl_s: float = 30.0
    # LRU de embeddings de query (task="query"): la misma pregunta repetida no
    # vuelve a pagar el roundtrip LiteLLM→Ollama. 0 = off.
    embed_query_cache_size: int = 128

    # RAG
    qdrant_host: str = "http://qdrant:6333"
    rag_collection: str = "llmops_docs"
    rag_embed_dim: int = 768
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 150
    rag_top_k: int = 5
    rag_docs_dir: str = "/app/docs"
    rag_watcher_enabled: bool = True
    # Búsqueda híbrida (denso + BM25 fusionados con RRF). El léxico rescata
    # queries con términos exactos/meta que el embedding difumina.
    rag_hybrid_enabled: bool = True
    rag_candidate_k: int = 20       # candidatos por rama antes de fusionar
    rag_rrf_k: int = 60             # constante RRF (amortigua ranks bajos)
    rag_bm25_weight: float = 1.5    # peso de la rama léxica en la fusión
    rag_dense_weight: float = 1.0   # peso de la rama densa

    # Observabilidad
    # Langfuse self-hosted (v2). Si langfuse_enabled=false o falta secret_key, no se traza.
    langfuse_enabled: bool = True
    langfuse_host: str = "http://langfuse:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    # Si true, mete prompt y completion en spans. En producción puede que quieras
    # apagarlo por privacidad — aquí lo dejamos por defecto on porque es entorno local.
    langfuse_log_payloads: bool = True
    # Prometheus
    metrics_enabled: bool = True

    # Web tools
    brave_api_key: str = os.getenv("BRAVE_API_KEY", "")
    brave_endpoint: str = "https://api.search.brave.com/res/v1/web/search"
    web_search_timeout: float = 10.0
    web_search_max_results: int = 10
    http_fetch_timeout: float = 10.0
    http_fetch_max_bytes: int = 2_000_000
    http_fetch_max_chars: int = 8000
    http_fetch_user_agent: str = "llmops-agent/0.1 (+https://github.com/local)"

    # Agent compute/system tools
    # Jail compartido por la tool filesystem y como cwd de python_exec. Ninguna
    # tool sale de aquí: el acceso se resuelve con path.resolve() + is_relative_to.
    tools_sandbox_dir: str = "/app/sandbox"
    fs_max_read_bytes: int = 256_000
    fs_max_write_bytes: int = 256_000
    # python_exec: subproceso Python aislado con rlimits (CPU/memoria/ficheros) +
    # timeout wall-clock. Aísla recursos y accidentes, NO es sandbox de seguridad
    # contra código adversarial (eso exigiría network namespace / contenedor).
    python_exec_timeout: float = 10.0
    python_exec_max_output: int = 8000
    python_exec_max_memory_mb: int = 512

    @property
    def api_keys_list(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]

    @property
    def bench_models_list(self) -> list[str]:
        return [m.strip() for m in self.bench_models.split(",") if m.strip()]


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
