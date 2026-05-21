from typing import Any, List, Tuple

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_pinecone import PineconeVectorStore

from app.config import RAG_FETCH_K_MAX, RAG_MMR_LAMBDA
from app.services.pinecone_store import (
    delete_namespace,
    fetch_documents_page,
    get_raw_index,
    namespace_has_vectors,
    pinecone_namespace,
)
from app.services.provider_service import Provider, get_embedding_model


def _load_pinecone_store(book_id: str, embedding_provider: Provider) -> VectorStore:
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


def load_book_store(book_id: str, embedding_provider: Provider = "openai") -> VectorStore:
    return _load_pinecone_store(book_id, embedding_provider)


def index_exists(book_id: str, embedding_provider: Provider) -> bool:
    return namespace_has_vectors(
        embedding_provider, pinecone_namespace(book_id, embedding_provider)
    )


def clear_book_index_vectors(book_id: str, embedding_provider: Provider) -> None:
    delete_namespace(embedding_provider, pinecone_namespace(book_id, embedding_provider))


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
    fk = fetch_k if fetch_k is not None else _fetch_k_for_k(k)
    lm = RAG_MMR_LAMBDA if lambda_mult is None else lambda_mult
    return store.max_marginal_relevance_search(query, k=k, fetch_k=fk, lambda_mult=lm)


def retrieve(
    book_id: str, query: str, k: int = 8, embedding_provider: Provider = "openai"
) -> List[Document]:
    store = load_book_store(book_id, embedding_provider=embedding_provider)
    return retrieve_from_store(store, query, k=k)


def list_book_documents_page(
    book_id: str,
    embedding_provider: Provider = "openai",
    *,
    offset: int = 0,
    limit: int = 50,
) -> Tuple[list[dict[str, Any]], int]:
    if limit < 1 or limit > 200:
        raise ValueError("limit must be between 1 and 200")
    if offset < 0:
        raise ValueError("offset must be non-negative")

    ns = pinecone_namespace(book_id, embedding_provider)
    return fetch_documents_page(
        embedding_provider,
        ns,
        offset=offset,
        limit=limit,
    )
