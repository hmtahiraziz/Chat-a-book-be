from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.config import (
    GEMINI_API_KEY,
    GEMINI_CHAT_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    OLLAMA_EMBED_MODEL,
)
from app.gemini_embedding_model import resolve_gemini_embedding_model

Provider = Literal["ollama", "google"]


def _validate_google_key() -> None:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to .env for google provider.")


def get_embedding_model(provider: Provider):
    if provider == "google":
        _validate_google_key()
        model = resolve_gemini_embedding_model()
        return GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=GEMINI_API_KEY,
        )
    return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)


def get_chat_model(provider: Provider, temperature: float = 0.1):
    if provider == "google":
        _validate_google_key()
        return ChatGoogleGenerativeAI(
            model=GEMINI_CHAT_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=temperature,
        )
    return ChatOllama(model=OLLAMA_CHAT_MODEL, base_url=OLLAMA_BASE_URL, temperature=temperature)
