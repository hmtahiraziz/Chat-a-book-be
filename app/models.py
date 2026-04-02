from typing import Literal

from pydantic import BaseModel, Field

Provider = Literal["ollama", "google"]
ChatRole = Literal["user", "assistant"]


class IngestResponse(BaseModel):
    book_id: str
    filename: str
    pages: int
    chunks: int
    embedding_provider: Provider


class ChatHistoryTurn(BaseModel):
    role: ChatRole
    content: str = Field(..., max_length=16_000)


class ChatRequest(BaseModel):
    book_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    k: int = Field(default=8, ge=1, le=20)
    embedding_provider: Provider = "ollama"
    chat_provider: Provider = "ollama"
    history: list[ChatHistoryTurn] = Field(default_factory=list, max_length=24)


class ClassifyRequest(BaseModel):
    question: str = Field(..., min_length=1)
    chat_provider: Provider = "ollama"


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8_000)
    voice: str | None = Field(default=None, max_length=64)

