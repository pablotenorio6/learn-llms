"""Configuración de la aplicación, cargada desde variables de entorno."""

from __future__ import annotations

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

    # Ollama
    ollama_host: str = "http://ollama:11434"
    ollama_request_timeout: int = 600
    ollama_keep_alive: str = "10m"

    # Modelos
    default_chat_model: str = "llama3.1:8b-instruct-q4_K_M"
    default_embed_model: str = "nomic-embed-text"
    bench_models: str = Field(
        default="llama3.1:8b-instruct-q4_K_M,qwen2.5:7b-instruct-q4_K_M,phi3.5:3.8b"
    )

    # RAG
    qdrant_host: str = "http://qdrant:6333"
    rag_collection: str = "llmops_docs"
    rag_embed_dim: int = 768
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 150
    rag_top_k: int = 5
    rag_docs_dir: str = "/app/docs"
    rag_watcher_enabled: bool = True

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
