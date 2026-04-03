from typing import Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

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


class CreatePineconeIndexRequest(BaseModel):
    """Create a serverless dense index (cosine) via the Pinecone control API."""

    name: str = Field(..., min_length=2, max_length=80)
    dimension: int | None = Field(
        default=None,
        ge=8,
        le=20_000,
        description="Vector size; omit if using preset.",
    )
    preset: Literal["google", "ollama"] | None = Field(
        default=None,
        description="google -> 3072, ollama -> 768 (typical nomic-embed-text).",
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
            raise ValueError("Provide either dimension or preset (google|ollama).")
        if self.dimension is not None and self.preset is not None:
            raise ValueError("Provide only one of dimension or preset.")
        return self

    def effective_dimension(self) -> int:
        if self.preset == "google":
            return 3072
        if self.preset == "ollama":
            return 768
        assert self.dimension is not None
        return self.dimension

