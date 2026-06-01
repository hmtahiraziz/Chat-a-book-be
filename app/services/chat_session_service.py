"""Chat threads persisted in MongoDB (scoped by client_id)."""

from __future__ import annotations

import time
import uuid
from typing import Any

from app.config import MONGODB_CHAT_SESSIONS_COLLECTION
from app.services.book_service import _db


def _sessions():
    return _db()[MONGODB_CHAT_SESSIONS_COLLECTION]


def _msg_from_api(msg: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(msg["id"]),
        "role": msg["role"],
        "content": str(msg["content"]),
        "classification": msg.get("classification"),
        "sources": msg.get("sources"),
        "created_at": int(msg.get("createdAt") or msg.get("created_at") or time.time() * 1000),
    }


def _msg_to_api(msg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": msg["id"],
        "role": msg["role"],
        "content": msg["content"],
        "createdAt": msg["created_at"],
    }
    if msg.get("classification") is not None:
        out["classification"] = msg["classification"]
    if msg.get("sources") is not None:
        out["sources"] = msg["sources"]
    return out


def _to_api(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": doc["id"],
        "bookId": doc["book_id"],
        "bookLabel": doc.get("book_label", ""),
        "embeddingProvider": doc.get("embedding_provider", "openai"),
        "chatProvider": doc.get("chat_provider", "openai"),
        "title": doc.get("title", "New chat"),
        "messages": [_msg_to_api(m) for m in doc.get("messages", [])],
        "updatedAt": doc.get("updated_at", 0),
    }


def _from_api(client_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages") or []
    return {
        "id": str(payload.get("id") or uuid.uuid4()),
        "client_id": client_id,
        "book_id": str(payload.get("bookId") or payload.get("book_id", "")),
        "book_label": str(payload.get("bookLabel") or payload.get("book_label", "")),
        "embedding_provider": payload.get("embeddingProvider")
        or payload.get("embedding_provider")
        or "openai",
        "chat_provider": payload.get("chatProvider") or payload.get("chat_provider") or "openai",
        "title": str(payload.get("title") or "New chat"),
        "messages": [_msg_from_api(m) for m in messages if isinstance(m, dict)],
        "updated_at": int(payload.get("updatedAt") or payload.get("updated_at") or time.time() * 1000),
    }


def list_sessions(client_id: str) -> list[dict[str, Any]]:
    cursor = _sessions().find({"client_id": client_id}).sort("updated_at", -1)
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for doc in cursor:
        session_id = doc.get("id")
        if not session_id:
            continue
        if session_id in seen:
            _sessions().delete_one({"_id": doc["_id"]})
            continue
        seen.add(session_id)
        out.append(_to_api(doc))
    return out


def get_session(client_id: str, session_id: str) -> dict[str, Any] | None:
    doc = _sessions().find_one({"client_id": client_id, "id": session_id})
    return _to_api(doc) if doc else None


def create_session(client_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    record = _from_api(client_id, payload)
    if not record["book_id"]:
        raise ValueError("bookId is required")
    _sessions().replace_one(
        {"client_id": client_id, "id": record["id"]},
        record,
        upsert=True,
    )
    return _to_api(record)


def replace_session(client_id: str, session_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    existing = _sessions().find_one({"client_id": client_id, "id": session_id})
    if not existing:
        return None
    record = _from_api(client_id, {**payload, "id": session_id})
    _sessions().replace_one({"client_id": client_id, "id": session_id}, record)
    return _to_api(record)


def delete_session(client_id: str, session_id: str) -> bool:
    result = _sessions().delete_one({"client_id": client_id, "id": session_id})
    return result.deleted_count > 0


def delete_sessions_by_book_id(book_id: str) -> int:
    result = _sessions().delete_many({"book_id": book_id})
    return result.deleted_count
