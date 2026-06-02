#!/usr/bin/env python3
"""Seed the shared demo account (no signup required — use POST /auth/demo-login).

Usage (from ai-book-chatbot-v2/):
  python scripts/seed_demo.py

Environment:
  DEMO_USER_EMAIL  (default: demo@bookchat.local)
  DEMO_USER_NAME   (default: Demo Guest)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from app.config import DEMO_USER_EMAIL, require_mongodb_config
from app.services.demo_service import resolve_demo_book
from app.services.user_service import create_user, get_user_by_email, subscribe_user


def main() -> None:
    require_mongodb_config()

    email = DEMO_USER_EMAIL.strip().lower()
    name = os.getenv("DEMO_USER_NAME", "Demo Guest").strip()

    existing = get_user_by_email(email)
    if existing:
        subscribe_user(existing["user_id"], "pro")
        print(f"Demo user already exists: {email} (Pro subscription refreshed)")
    else:
        doc = create_user(
            email=email,
            password=os.getenv("DEMO_USER_PASSWORD", "demo-not-used-for-login"),
            name=name,
            role="user",
            plan_id="pro",
            subscription_status="active",
        )
        print("Demo user created:")
        print(f"  email: {doc['email']}")
        print("  login: POST /auth/demo-login (no password)")

    book_id, label = resolve_demo_book()
    if book_id:
        print(f"  demo book: {label!r} ({book_id})")
    else:
        print("  demo book: not found — ingest Harry Potter PDF or set DEMO_BOOK_ID in .env")


if __name__ == "__main__":
    main()
