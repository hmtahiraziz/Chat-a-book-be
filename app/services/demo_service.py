"""Shared demo account and demo book resolution."""

from __future__ import annotations

from typing import Any

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
