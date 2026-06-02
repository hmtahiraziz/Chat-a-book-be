"""Shared demo account and demo book resolution."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from app.config import DEMO_BOOK_ID, DEMO_USER_EMAIL
from app.services.book_service import get_book, list_books
from app.services.user_service import (
    get_user_by_email,
    has_active_subscription,
    subscribe_user,
)


def resolve_demo_book() -> tuple[str | None, str | None]:
    """Return (book_id, display_label) for the Harry Potter demo book if present."""
    if DEMO_BOOK_ID:
        entry = get_book(DEMO_BOOK_ID)
        if entry:
            return DEMO_BOOK_ID, str(entry.get("filename") or DEMO_BOOK_ID)

    for doc in list_books().values():
        filename = str(doc.get("filename") or "").lower()
        book_id = str(doc.get("book_id") or "")
        haystack = f"{filename} {book_id}".lower()
        if "harry" in haystack and "potter" in haystack:
            return book_id, str(doc.get("filename") or book_id)

    return None, None


def is_demo_user(user: dict[str, Any]) -> bool:
    return (user.get("email") or "").strip().lower() == DEMO_USER_EMAIL


def demo_book_id_or_none() -> str | None:
    book_id, _ = resolve_demo_book()
    return book_id


def filter_books_for_user(user: dict[str, Any], books: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not is_demo_user(user):
        return books
    demo_id = demo_book_id_or_none()
    if not demo_id:
        return []
    return [b for b in books if b.get("book_id") == demo_id]


def enforce_demo_book_access(user: dict[str, Any], book_id: str) -> None:
    if not is_demo_user(user):
        return
    demo_id = demo_book_id_or_none()
    if not demo_id or book_id != demo_id:
        raise HTTPException(
            status_code=403,
            detail="Demo accounts can only access the sample Harry Potter book.",
        )


def enforce_not_demo_user(user: dict[str, Any], *, action: str = "perform this action") -> None:
    if is_demo_user(user):
        raise HTTPException(
            status_code=403,
            detail=f"Demo accounts cannot {action}. Create a full account to upload and manage books.",
        )


def ensure_demo_user() -> dict[str, Any]:
    doc = get_user_by_email(DEMO_USER_EMAIL)
    if not doc:
        raise RuntimeError(
            f"Demo user {DEMO_USER_EMAIL!r} not found. Run: python scripts/seed_demo.py"
        )
    if not has_active_subscription(doc):
        updated = subscribe_user(doc["user_id"], "pro")
        if updated:
            doc = get_user_by_email(DEMO_USER_EMAIL) or doc
    return doc
