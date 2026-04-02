"""Pinecone index helpers: namespaces, upsert with precomputed embeddings, listing, delete."""

from __future__ import annotations

import json
import re
from typing import Any, List

from langchain_core.documents import Document
from pinecone import NotFoundException, Pinecone

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
    idx.upsert(vectors=vectors, namespace=namespace, show_progress=False)


def delete_namespace(provider: Provider, namespace: str) -> None:
    """Remove all vectors in a namespace. No-op if the Pinecone index does not exist."""
    try:
        idx = get_raw_index(provider)
        idx.delete(delete_all=True, namespace=namespace)
    except NotFoundException:
        # Index missing (renamed, deleted, or wrong PINECONE_INDEX_* env); vectors are already gone.
        return


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
