from typing import Literal

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import (
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
    OPENAI_EMBED_MODEL,
)

Provider = Literal["openai"]


def _validate_openai_key() -> None:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env.")


def get_embedding_model(provider: Provider = "openai"):
    _validate_openai_key()
    return OpenAIEmbeddings(
        model=OPENAI_EMBED_MODEL,
        api_key=OPENAI_API_KEY,
    )


def get_chat_model(provider: Provider = "openai", temperature: float = 0.1):
    _validate_openai_key()
    return ChatOpenAI(
        model=OPENAI_CHAT_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=temperature,
    )
