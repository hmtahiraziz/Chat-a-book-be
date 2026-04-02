"""Resolve a working Gemini embedding model id for the user's API key."""

from __future__ import annotations

from typing import List

from google import genai

from app.config import GEMINI_API_KEY, GEMINI_EMBED_MODEL

_cached_model: str | None = None

_PREFERRED_ORDER: List[str] = [
    "gemini-embedding-001",
    "gemini-embedding-2-preview",
    "text-embedding-004",
    "embedding-001",
]


def _normalize(name: str) -> str:
    n = name.strip()
    if n.startswith("models/"):
        n = n[len("models/") :]
    return n


def _supports_embed(model: object) -> bool:
    actions = getattr(model, "supported_actions", None) or getattr(
        model, "supported_generation_methods", None
    ) or []
    if isinstance(actions, (list, tuple, set)):
        return "embedContent" in actions
    return "embedContent" in str(actions)


def resolve_gemini_embedding_model() -> str:
    """Return a short model name suitable for GoogleGenerativeAIEmbeddings (e.g. gemini-embedding-001)."""
    global _cached_model
    if _cached_model is not None:
        return _cached_model
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    preferred = _normalize(GEMINI_EMBED_MODEL)
    client = genai.Client(api_key=GEMINI_API_KEY)

    available_embed: list[str] = []
    for m in client.models.list(config={"page_size": 200}):
        if not _supports_embed(m):
            continue
        raw = getattr(m, "name", "") or ""
        available_embed.append(_normalize(str(raw)))

    if not available_embed:
        raise RuntimeError(
            "No Gemini embedding models are available for this API key. "
            "Check billing and API access, or use embedding_provider=ollama."
        )

    # Exact match on preferred (short name).
    if preferred and preferred in available_embed:
        _cached_model = preferred
        return _cached_model

    # Try preferred order against API list.
    for candidate in _PREFERRED_ORDER:
        if candidate in available_embed:
            _cached_model = candidate
            return _cached_model

    # First API-listed embedding model.
    _cached_model = available_embed[0]
    return _cached_model


def reset_gemini_embedding_model_cache() -> None:
    global _cached_model
    _cached_model = None
