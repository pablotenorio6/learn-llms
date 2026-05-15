"""Schemas Pydantic compatibles con la API de OpenAI.

Solo cubren lo que la Fase 1 necesita: chat completions, embeddings, models.
Tools/function calling llegará en Fase 4.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---- Chat ----

Role = Literal["system", "user", "assistant", "tool"]


class ChatMessage(BaseModel):
    role: Role
    content: str
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1)
    stream: bool = False
    stop: list[str] | str | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    user: str | None = None  # ignorado, solo para compat


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] | None = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage


# ---- Streaming chunks ----

class ChatChoiceDelta(BaseModel):
    role: Role | None = None
    content: str | None = None


class ChatChunkChoice(BaseModel):
    index: int = 0
    delta: ChatChoiceDelta
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatChunkChoice]


# ---- Embeddings ----

class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: Literal["float", "base64"] = "float"
    user: str | None = None


class EmbeddingItem(BaseModel):
    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingItem]
    model: str
    usage: Usage


# ---- Models ----

class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "litellm"
    # extra info (no estándar OpenAI pero útil)
    metadata: dict[str, Any] | None = None


class ModelsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]


# ---- RAG ----

class RagDocument(BaseModel):
    doc_id: str
    source: str
    chunks: int


class RagDocumentsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[RagDocument]


class RagQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class RagHit(BaseModel):
    text: str
    score: float
    source: str
    doc_id: str
    chunk_idx: int


class RagQueryResponse(BaseModel):
    query: str
    hits: list[RagHit]
    system_message: str


class RagIndexResponse(BaseModel):
    doc_id: str
    source: str
    chunks_indexed: int
    bytes: int


# ---- Agents ----

class AgentRunRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    tools_allowed: list[str] | None = None      # None = todas las registradas
    max_iterations: int = Field(default=10, ge=1, le=50)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


class AgentToolInfo(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class AgentToolsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[AgentToolInfo]
