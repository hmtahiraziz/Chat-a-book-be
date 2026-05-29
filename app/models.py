from typing import Any, Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

Provider = Literal["openai"]
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
    embedding_provider: Provider = "openai"
    chat_provider: Provider = "openai"
    history: list[ChatHistoryTurn] = Field(default_factory=list, max_length=24)


class ChatMessagePayload(BaseModel):
    id: str = Field(..., min_length=1)
    role: ChatRole
    content: str = Field(..., max_length=32_000)
    classification: str | None = Field(default=None, max_length=256)
    sources: list[dict[str, Any]] | None = None
    createdAt: int = Field(default=0, ge=0)


class ChatSessionPayload(BaseModel):
    id: str | None = Field(default=None, max_length=128)
    bookId: str = Field(..., min_length=1)
    bookLabel: str = Field(default="", max_length=512)
    embeddingProvider: Provider = "openai"
    chatProvider: Provider = "openai"
    title: str = Field(default="New chat", max_length=256)
    messages: list[ChatMessagePayload] = Field(default_factory=list, max_length=200)
    updatedAt: int = Field(default=0, ge=0)


class ClassifyRequest(BaseModel):
    question: str = Field(..., min_length=1)
    chat_provider: Provider = "openai"


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8_000)
    voice: str | None = Field(default=None, max_length=64)


class CreatePineconeIndexRequest(BaseModel):
    """Create a serverless dense index (cosine) via the Pinecone control API."""

    name: str = Field(..., min_length=2, max_length=80)
    dimension: int | None = Field(
        default=None,
        ge=8,
        le=20_000,
        description="Vector size; omit if using preset.",
    )
    preset: Literal["openai"] | None = Field(
        default=None,
        description="openai -> 3072 (text-embedding-3-large).",
    )
    metric: Literal["cosine", "dotproduct", "euclidean"] = "cosine"
    cloud: str | None = Field(default=None, max_length=32)
    region: str | None = Field(default=None, max_length=32)

    @field_validator("name")
    @classmethod
    def normalize_index_name(cls, v: str) -> str:
        return v.strip().lower()

    @model_validator(mode="after")
    def dimension_or_preset(self) -> Self:
        if self.dimension is None and self.preset is None:
            raise ValueError("Provide either dimension or preset (openai).")
        if self.dimension is not None and self.preset is not None:
            raise ValueError("Provide only one of dimension or preset.")
        return self

    def effective_dimension(self) -> int:
        if self.preset == "openai":
            return 3072
        assert self.dimension is not None
        return self.dimension
