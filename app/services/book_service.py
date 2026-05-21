"""Book library metadata and PDF storage in MongoDB (GridFS)."""

from __future__ import annotations

from typing import Any

from bson import ObjectId
from gridfs import GridFS
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from app.config import (
    MONGODB_BOOKS_COLLECTION,
    MONGODB_DB_NAME,
    MONGODB_PDF_BUCKET,
    MONGODB_URI,
)

_client: MongoClient | None = None


def _db() -> Database:
    global _client
    if _client is None:
        if not MONGODB_URI:
            raise RuntimeError("MONGODB_URI is not set. Add it to .env for book storage.")
        _client = MongoClient(MONGODB_URI)
    return _client[MONGODB_DB_NAME]


def _books() -> Collection:
    return _db()[MONGODB_BOOKS_COLLECTION]


def _gridfs() -> GridFS:
    return GridFS(_db(), collection=MONGODB_PDF_BUCKET)


def _serialize(doc: dict[str, Any]) -> dict[str, Any]:
    out = dict(doc)
    out.pop("_id", None)
    if "pdf_file_id" in out and out["pdf_file_id"] is not None:
        out["pdf_file_id"] = str(out["pdf_file_id"])
    return out


def upsert_book(book_id: str, payload: dict[str, Any]) -> None:
    record = {**payload, "book_id": book_id}
    _books().update_one({"book_id": book_id}, {"$set": record}, upsert=True)


def list_books() -> dict[str, dict[str, Any]]:
    return {doc["book_id"]: _serialize(doc) for doc in _books().find()}


def get_book(book_id: str) -> dict[str, Any] | None:
    doc = _books().find_one({"book_id": book_id})
    return _serialize(doc) if doc else None


def delete_book(book_id: str) -> dict[str, Any] | None:
    doc = _books().find_one({"book_id": book_id})
    if not doc:
        return None
    _books().delete_one({"book_id": book_id})
    return _serialize(doc)


def count_books_with_pdf_file_id(pdf_file_id: str | ObjectId, *, exclude_book_id: str | None = None) -> int:
    fid = ObjectId(pdf_file_id) if isinstance(pdf_file_id, str) else pdf_file_id
    query: dict[str, Any] = {"pdf_file_id": fid}
    if exclude_book_id:
        query["book_id"] = {"$ne": exclude_book_id}
    return _books().count_documents(query)


def store_pdf(base_book_id: str, content: bytes) -> str:
    """Store or replace PDF bytes for a base book id; returns GridFS file id as string."""
    fs = _gridfs()
    for existing in fs.find({"metadata.base_book_id": base_book_id}):
        fs.delete(existing._id)
    file_id = fs.put(
        content,
        filename=f"{base_book_id}.pdf",
        metadata={"base_book_id": base_book_id},
    )
    return str(file_id)


def read_pdf_bytes(pdf_file_id: str) -> bytes | None:
    try:
        oid = ObjectId(pdf_file_id)
    except Exception:
        return None
    grid_out = _gridfs().get(oid)
    return grid_out.read()


def delete_pdf_file(pdf_file_id: str) -> bool:
    try:
        oid = ObjectId(pdf_file_id)
    except Exception:
        return False
    fs = _gridfs()
    if not fs.exists(oid):
        return False
    fs.delete(oid)
    return True
