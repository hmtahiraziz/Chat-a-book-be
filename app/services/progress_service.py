"""Ingest resume checkpoints in MongoDB (replaces local progress JSON files)."""

from __future__ import annotations

from typing import Any

from app.config import MONGODB_INGEST_PROGRESS_COLLECTION
from app.services.book_service import _db


def _collection():
    return _db()[MONGODB_INGEST_PROGRESS_COLLECTION]


def load_progress(book_id: str) -> dict[str, Any]:
    doc = _collection().find_one({"book_id": book_id})
    if not doc:
        return {}
    return {k: v for k, v in doc.items() if k not in ("_id", "book_id")}


def save_progress(book_id: str, payload: dict[str, Any]) -> None:
    record = {**payload, "book_id": book_id}
    _collection().update_one({"book_id": book_id}, {"$set": record}, upsert=True)


def delete_progress(book_id: str) -> bool:
    result = _collection().delete_one({"book_id": book_id})
    return result.deleted_count > 0
