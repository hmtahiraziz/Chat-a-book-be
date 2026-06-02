#!/usr/bin/env python3
"""Seed an admin user with an active Pro subscription.

Usage (from ai-book-chatbot-v2/):
  python scripts/seed_admin.py

Environment:
  SEED_ADMIN_EMAIL     (default: admin@bookchat.local)
  SEED_ADMIN_PASSWORD  (default: Admin123! — change in production)
  SEED_ADMIN_NAME      (default: Admin)
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

from app.config import require_mongodb_config
from app.services.user_service import create_user, get_user_by_email, subscribe_user


def main() -> None:
    require_mongodb_config()

    email = os.getenv("SEED_ADMIN_EMAIL", "admin@bookchat.local").strip().lower()
    password = os.getenv("SEED_ADMIN_PASSWORD", "Admin123!")
    name = os.getenv("SEED_ADMIN_NAME", "Admin").strip()

    existing = get_user_by_email(email)
    if existing:
        subscribe_user(existing["user_id"], "pro")
        print(f"Admin already exists: {email} (subscription refreshed to Pro)")
        return

    doc = create_user(
        email=email,
        password=password,
        name=name,
        role="admin",
        plan_id="pro",
        subscription_status="active",
    )
    print("Admin user created:")
    print(f"  email:    {doc['email']}")
    print(f"  password: {password}")
    print(f"  role:     {doc['role']}")
    print(f"  plan:     pro (active)")


if __name__ == "__main__":
    main()
