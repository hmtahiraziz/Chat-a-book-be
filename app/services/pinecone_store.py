"""Pinecone index helpers: namespaces, upsert with precomputed embeddings, listing, delete."""

from __future__ import annotations

import json
import re
from typing import Any, List

from langchain_core.documents import Document
from pinecone import NotFoundException, Pinecone, ServerlessSpec

from app.config import PINECONE_API_KEY, pinecone_index_name_for_provider
from app.services.provider_service import Provider

_pinecone_client: Pinecone | None = None
_index_cache: dict[str, Any] = {}


def _pc() -> Pinecone:
    global _pinecone_client
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is not set.")
    if _pinecone_client is None:
        _pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
    return _pinecone_client


def get_raw_index(provider: Provider) -> Any:
    name = pinecone_index_name_for_provider(provider)
    if not name:
        raise RuntimeError("PINECONE_INDEX (or provider-specific index name) is not set.")
    if name not in _index_cache:
        _index_cache[name] = _pc().Index(name)
    return _index_cache[name]


def pinecone_namespace(book_id: str, embedding_provider: Provider) -> str:
    """Pinecone namespace: alphanumeric, dash, underscore; max 512 chars."""
    raw = f"{embedding_provider}__{book_id}"
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", raw)
    return safe[:512]


def chunk_id(global_ordinal: int) -> str:
    return f"c{global_ordinal:08d}"


def _parse_pinecone_dimension_mismatch(exc: BaseException) -> tuple[str | None, str | None]:
    """Returns (embedding_dim, index_dim) from Pinecone gRPC error text, if present."""
    raw = str(exc)
    m = re.search(
        r"Vector dimension (\d+).*?dimension of the index (\d+)",
        raw,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m.group(1), m.group(2)
    return None, None


def _sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[str(k)] = v
        elif isinstance(v, list) and all(isinstance(x, str) for x in v):
            out[str(k)] = v
        else:
            out[str(k)] = json.dumps(v) if not isinstance(v, str) else v
    return out


def upsert_embedding_batch(
    provider: Provider,
    namespace: str,
    global_start: int,
    texts: List[str],
    embeddings: List[List[float]],
    metadatas: List[dict[str, Any]],
    *,
    text_key: str = "text",
) -> None:
    """Upsert one batch using precomputed embedding vectors (matches ingest pipeline)."""
    idx = get_raw_index(provider)
    if len(texts) != len(embeddings) or len(texts) != len(metadatas):
        raise ValueError("texts, embeddings, and metadatas must have the same length.")
    vectors: list = []
    for i, (text, emb, meta) in enumerate(zip(texts, embeddings, metadatas)):
        cid = chunk_id(global_start + i)
        m = dict(meta)
        m[text_key] = text
        m["chunk_ordinal"] = int(global_start + i)
        m = _sanitize_metadata(m)
        vectors.append((cid, emb, m))
    try:
        idx.upsert(vectors=vectors, namespace=namespace, show_progress=False)
    except Exception as exc:
        msg = str(exc).lower()
        if "dimension" in msg and ("does not match" in msg or "mismatch" in msg):
            index_name = pinecone_index_name_for_provider(provider)
            emb_d, idx_d = _parse_pinecone_dimension_mismatch(exc)
            if emb_d and idx_d:
                hint = (
                    f"Pinecone index {index_name!r} was created with dimension {idx_d}, but your "
                    f"embeddings are dimension {emb_d}. The index name in Pinecone does not control "
                    f"its size—in the Pinecone console, delete this index and create a new one with "
                    f"dimension {emb_d} (metric cosine), then keep PINECONE_INDEX_GOOGLE={index_name!r} "
                    f"in .env or point it at the new index name."
                )
            elif provider == "google":
                hint = (
                    f"Pinecone index {index_name!r} does not match Google embedding size "
                    f"(gemini-embedding-001 uses 3072). Recreate the index at 3072 dimensions or fix "
                    f"PINECONE_INDEX_GOOGLE in .env (see .env.example)."
                )
            else:
                hint = (
                    f"Pinecone index {index_name!r} does not match your Ollama embed model dimensions "
                    f"(e.g. nomic-embed-text uses 768). Recreate a matching index and set "
                    f"PINECONE_INDEX_OLLAMA (or PINECONE_INDEX) in .env."
                )
            raise RuntimeError(hint) from exc
        raise


def delete_namespace(provider: Provider, namespace: str) -> None:
    """Remove all vectors in a namespace. No-op if the Pinecone index does not exist."""
    try:
        idx = get_raw_index(provider)
        idx.delete(delete_all=True, namespace=namespace)
    except NotFoundException:
        # Index missing (renamed, deleted, or wrong PINECONE_INDEX_* env); vectors are already gone.
        return


def create_serverless_pinecone_index(
    name: str,
    dimension: int,
    *,
    metric: str = "cosine",
    cloud: str = "aws",
    region: str = "us-east-1",
) -> dict[str, Any]:
    """Create a serverless dense index via the Pinecone control API."""
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is not set.")
    pc = _pc()
    spec = ServerlessSpec(cloud=cloud, region=region)
    model = pc.create_index(
        name=name,
        spec=spec,
        dimension=dimension,
        metric=metric,
    )
    return model.to_dict()


def namespace_has_vectors(provider: Provider, namespace: str) -> bool:
    """True if the namespace has at least one vector (cheap `list` probe)."""
    idx = get_raw_index(provider)
    res = idx.list_paginated(namespace=namespace, limit=1)
    return len(getattr(res, "vectors", None) or []) > 0


def list_all_ids(provider: Provider, namespace: str) -> List[str]:
    idx = get_raw_index(provider)
    out: List[str] = []
    for batch in idx.list(namespace=namespace):
        out.extend(batch)
    return out


def fetch_documents_page(
    provider: Provider,
    namespace: str,
    *,
    offset: int,
    limit: int,
    text_key: str = "text",
) -> tuple[list[dict[str, Any]], int]:
    """List chunks by listing IDs, fetch metadata, sort by chunk_ordinal, paginate."""
    all_ids = list_all_ids(provider, namespace)
    if not all_ids:
        return [], 0

    idx = get_raw_index(provider)
    fetched: list[tuple[int, str, Document]] = []

    batch_size = 100
    for i in range(0, len(all_ids), batch_size):
        batch = all_ids[i : i + batch_size]
        res = idx.fetch(ids=batch, namespace=namespace)
        vecs = res.vectors or {}
        for vid in batch:
            v = vecs.get(vid)
            if v is None:
                continue
            meta = dict(v.metadata) if v.metadata else {}
            text = meta.get(text_key, "")
            ord_key = meta.get("chunk_ordinal")
            try:
                ordinal = int(ord_key) if ord_key is not None else -1
            except (TypeError, ValueError):
                ordinal = -1
            doc_meta = {k: v for k, v in meta.items() if k not in (text_key, "chunk_ordinal")}
            doc = Document(page_content=str(text), metadata=doc_meta)
            fetched.append((ordinal, vid, doc))

    fetched.sort(key=lambda x: (x[0] if x[0] >= 0 else 10**9, x[1]))
    total = len(fetched)
    window = fetched[offset : offset + limit]
    chunks: list[dict[str, Any]] = []
    for rank, (ordinal, vid, doc) in enumerate(window):
        faiss_like = ordinal if ordinal >= 0 else rank
        chunks.append(
            {
                "ordinal": offset + rank + 1,
                "faiss_index": faiss_like,
                "doc_id": vid,
                "text": doc.page_content,
                "metadata": dict(doc.metadata),
                "char_count": len(doc.page_content),
            }
        )
    return chunks, total
