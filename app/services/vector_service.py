from pathlib import Path
from typing import Any, List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from app.config import INDEX_DIR, RAG_FETCH_K_MAX, RAG_MMR_LAMBDA, use_pinecone_vector_store
from app.services.provider_service import Provider, get_embedding_model


def _provider_book_id(book_id: str, provider: Provider) -> str:
    return f"{provider}__{book_id}"


# --- FAISS (local) ---


def faiss_index_dir(book_id: str, embedding_provider: Provider) -> Path:
    return INDEX_DIR / _provider_book_id(book_id, embedding_provider)


def faiss_index_exists(book_id: str, embedding_provider: Provider) -> bool:
    folder = faiss_index_dir(book_id, embedding_provider)
    return (folder / "index.faiss").exists() and (folder / "index.pkl").exists()


def index_book(book_id: str, docs: List[Document], embedding_provider: Provider = "ollama") -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store = FAISS.from_documents(docs, get_embedding_model(embedding_provider))
    store.save_local(str(INDEX_DIR / _provider_book_id(book_id, embedding_provider)))


def _load_faiss_store(book_id: str, embedding_provider: Provider) -> FAISS:
    path = INDEX_DIR / _provider_book_id(book_id, embedding_provider)
    if not path.exists():
        raise FileNotFoundError(
            f"Index not found for book: {book_id} (embedding_provider={embedding_provider})"
        )
    return FAISS.load_local(
        str(path),
        get_embedding_model(embedding_provider),
        allow_dangerous_deserialization=True,
    )


# --- Pinecone ---


def _load_pinecone_store(book_id: str, embedding_provider: Provider) -> VectorStore:
    from langchain_pinecone import PineconeVectorStore

    from app.services.pinecone_store import get_raw_index, namespace_has_vectors, pinecone_namespace

    if not use_pinecone_vector_store():
        raise RuntimeError("Pinecone is not configured.")
    ns = pinecone_namespace(book_id, embedding_provider)
    if not namespace_has_vectors(embedding_provider, ns):
        raise FileNotFoundError(
            f"Index not found for book: {book_id} (embedding_provider={embedding_provider})"
        )
    idx = get_raw_index(embedding_provider)
    return PineconeVectorStore(
        index=idx,
        embedding=get_embedding_model(embedding_provider),
        namespace=ns,
    )


def load_book_store(book_id: str, embedding_provider: Provider = "ollama") -> VectorStore:
    if use_pinecone_vector_store():
        return _load_pinecone_store(book_id, embedding_provider)
    return _load_faiss_store(book_id, embedding_provider)


def index_exists(book_id: str, embedding_provider: Provider) -> bool:
    if use_pinecone_vector_store():
        from app.services.pinecone_store import namespace_has_vectors, pinecone_namespace

        return namespace_has_vectors(embedding_provider, pinecone_namespace(book_id, embedding_provider))
    return faiss_index_exists(book_id, embedding_provider)


def clear_book_index_files(book_id: str, embedding_provider: Provider) -> None:
    """Remove local FAISS folder (no-op for Pinecone)."""
    if use_pinecone_vector_store():
        return
    folder = faiss_index_dir(book_id, embedding_provider)
    if folder.exists():
        import shutil

        shutil.rmtree(folder, ignore_errors=True)


def clear_book_index_vectors(book_id: str, embedding_provider: Provider) -> None:
    """Delete all vectors for a book (Pinecone namespace or FAISS folder)."""
    if use_pinecone_vector_store():
        from app.services.pinecone_store import delete_namespace, pinecone_namespace

        delete_namespace(embedding_provider, pinecone_namespace(book_id, embedding_provider))
    else:
        clear_book_index_files(book_id, embedding_provider)


def _fetch_k_for_k(k: int) -> int:
    return min(max(k * 4, 16), RAG_FETCH_K_MAX)


def retrieve_from_store(
    store: VectorStore,
    query: str,
    k: int = 8,
    *,
    fetch_k: int | None = None,
    lambda_mult: float | None = None,
) -> List[Document]:
    """MMR-based retrieval for diverse, relevant chunks."""
    fk = fetch_k if fetch_k is not None else _fetch_k_for_k(k)
    lm = RAG_MMR_LAMBDA if lambda_mult is None else lambda_mult
    return store.max_marginal_relevance_search(query, k=k, fetch_k=fk, lambda_mult=lm)


def retrieve(
    book_id: str, query: str, k: int = 8, embedding_provider: Provider = "ollama"
) -> List[Document]:
    store = load_book_store(book_id, embedding_provider=embedding_provider)
    return retrieve_from_store(store, query, k=k)


def list_book_documents_page(
    book_id: str,
    embedding_provider: Provider = "ollama",
    *,
    offset: int = 0,
    limit: int = 50,
) -> Tuple[list[dict[str, Any]], int]:
    """Human-readable chunk listing (FAISS docstore or Pinecone metadata)."""
    if limit < 1 or limit > 200:
        raise ValueError("limit must be between 1 and 200")
    if offset < 0:
        raise ValueError("offset must be non-negative")

    if use_pinecone_vector_store():
        from app.services.pinecone_store import fetch_documents_page, pinecone_namespace

        ns = pinecone_namespace(book_id, embedding_provider)
        return fetch_documents_page(
            embedding_provider,
            ns,
            offset=offset,
            limit=limit,
        )

    store = _load_faiss_store(book_id, embedding_provider=embedding_provider)
    ordered: list[tuple[int, str, Document]] = []
    for faiss_idx in sorted(store.index_to_docstore_id.keys()):
        doc_id = store.index_to_docstore_id[faiss_idx]
        doc = store.docstore.search(doc_id)
        if isinstance(doc, Document):
            ordered.append((faiss_idx, doc_id, doc))

    total = len(ordered)
    window = ordered[offset : offset + limit]
    chunks: list[dict[str, Any]] = []
    for rank, (faiss_idx, doc_id, doc) in enumerate(window):
        chunks.append(
            {
                "ordinal": offset + rank + 1,
                "faiss_index": faiss_idx,
                "doc_id": doc_id,
                "text": doc.page_content,
                "metadata": dict(doc.metadata),
                "char_count": len(doc.page_content),
            }
        )
    return chunks, total
