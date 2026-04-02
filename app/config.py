import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BOOKS_DIR = DATA_DIR / "books"
INDEX_DIR = DATA_DIR / "indices"
PROGRESS_DIR = DATA_DIR / "progress"
MANIFEST_FILE = DATA_DIR / "manifest.json"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
GEMINI_TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts")
GEMINI_TTS_VOICE = os.getenv("GEMINI_TTS_VOICE", "charon")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# RAG retrieval: MMR over a wider candidate pool for more diverse context
RAG_FETCH_K_MAX = int(os.getenv("RAG_FETCH_K_MAX", "48"))
RAG_MMR_LAMBDA = float(os.getenv("RAG_MMR_LAMBDA", "0.55"))

# If set, admin routes require header: X-Admin-Token: <value>
ADMIN_API_TOKEN = os.getenv("ADMIN_API_TOKEN", "").strip()

# Vector store: "faiss" (local disk) or "pinecone" (hosted).
# If VECTOR_STORE is unset: use Pinecone when PINECONE_API_KEY and PINECONE_INDEX are set, else FAISS.
_VECTOR_STORE_RAW = os.getenv("VECTOR_STORE", "").strip().lower()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "").strip()
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "").strip()
PINECONE_INDEX_OLLAMA = os.getenv("PINECONE_INDEX_OLLAMA", "").strip()
PINECONE_INDEX_GOOGLE = os.getenv("PINECONE_INDEX_GOOGLE", "").strip()


def use_pinecone_vector_store() -> bool:
    if _VECTOR_STORE_RAW == "pinecone":
        return True
    if _VECTOR_STORE_RAW == "faiss":
        return False
    return bool(PINECONE_API_KEY and PINECONE_INDEX)


def pinecone_index_name_for_provider(provider: str) -> str:
    """Separate Pinecone indexes per embedding provider if dimensions differ (768 vs 3072)."""
    if provider == "google" and PINECONE_INDEX_GOOGLE:
        return PINECONE_INDEX_GOOGLE
    if provider == "ollama" and PINECONE_INDEX_OLLAMA:
        return PINECONE_INDEX_OLLAMA
    return PINECONE_INDEX


def public_vector_store_info() -> dict[str, Any]:
    """Safe to expose to the UI: no API keys."""
    resolved = "pinecone" if use_pinecone_vector_store() else "faiss"
    env_mode = _VECTOR_STORE_RAW or "auto"
    payload: dict[str, Any] = {
        "vector_store": resolved,
        "vector_store_env": env_mode,
        "vector_store_label": (
            "Pinecone (hosted)"
            if resolved == "pinecone"
            else "FAISS (local files on the API server)"
        ),
    }
    if resolved == "pinecone":
        payload["pinecone_indexes"] = {
            "default": PINECONE_INDEX or None,
            "ollama": PINECONE_INDEX_OLLAMA or None,
            "google": PINECONE_INDEX_GOOGLE or None,
        }
    return payload
