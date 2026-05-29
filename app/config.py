import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "tts-1")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# RAG retrieval: MMR over a wider candidate pool for more diverse context
RAG_FETCH_K_MAX = int(os.getenv("RAG_FETCH_K_MAX", "48"))
RAG_MMR_LAMBDA = float(os.getenv("RAG_MMR_LAMBDA", "0.55"))

# If set, admin routes require header: X-Admin-Token: <value>
ADMIN_API_TOKEN = os.getenv("ADMIN_API_TOKEN", "").strip()

# MongoDB — book library metadata and PDFs (GridFS)
MONGODB_URI = os.getenv("MONGODB_URI", "").strip()
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "bookchat").strip()
MONGODB_BOOKS_COLLECTION = os.getenv("MONGODB_BOOKS_COLLECTION", "books").strip()
MONGODB_PDF_BUCKET = os.getenv("MONGODB_PDF_BUCKET", "pdfs").strip()
MONGODB_INGEST_PROGRESS_COLLECTION = os.getenv(
    "MONGODB_INGEST_PROGRESS_COLLECTION", "ingest_progress"
).strip()
MONGODB_CHAT_SESSIONS_COLLECTION = os.getenv(
    "MONGODB_CHAT_SESSIONS_COLLECTION", "chat_sessions"
).strip()

# Pinecone — required vector store for embeddings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "").strip()
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "").strip()
PINECONE_INDEX_OPENAI = os.getenv("PINECONE_INDEX_OPENAI", "").strip()

PINECONE_SERVERLESS_CLOUD = os.getenv("PINECONE_SERVERLESS_CLOUD", "aws").strip()
PINECONE_SERVERLESS_REGION = os.getenv("PINECONE_SERVERLESS_REGION", "us-east-1").strip()


def require_pinecone_config() -> None:
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is not set.")
    if not PINECONE_INDEX and not PINECONE_INDEX_OPENAI:
        raise RuntimeError(
            "Set PINECONE_INDEX (or PINECONE_INDEX_OPENAI) in .env."
        )


def require_mongodb_config() -> None:
    if not MONGODB_URI:
        raise RuntimeError("MONGODB_URI is not set.")


def pinecone_index_name_for_provider(provider: str) -> str:
    """Pinecone index for OpenAI embeddings (3072-dim text-embedding-3-large)."""
    if PINECONE_INDEX_OPENAI:
        return PINECONE_INDEX_OPENAI
    if PINECONE_INDEX:
        return PINECONE_INDEX
    raise RuntimeError("No Pinecone index configured. Set PINECONE_INDEX_OPENAI or PINECONE_INDEX in .env.")


def public_vector_store_info() -> dict[str, Any]:
    """Safe to expose to the UI: no API keys."""
    payload: dict[str, Any] = {
        "vector_store": "pinecone",
        "vector_store_label": "Pinecone (hosted)",
        "library_store": "mongodb",
        "library_store_label": "MongoDB (metadata + PDFs)",
        "pinecone_indexes": {
            "default": PINECONE_INDEX or None,
            "openai": PINECONE_INDEX_OPENAI or PINECONE_INDEX or None,
        },
        "pinecone_embedding_dimensions": {
            "openai": 3072,
            "note": "OpenAI size is for text-embedding-3-large default (3072).",
        },
    }
    if PINECONE_API_KEY:
        payload["pinecone_create_index_route"] = "POST /admin/pinecone/index"
        payload["pinecone_create_index_defaults"] = {
            "cloud": PINECONE_SERVERLESS_CLOUD,
            "region": PINECONE_SERVERLESS_REGION,
        }
    return payload
