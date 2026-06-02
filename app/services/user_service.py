"""User accounts and subscriptions in MongoDB."""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pymongo import ASCENDING
from pymongo.collection import Collection

from app.config import MONGODB_USERS_COLLECTION
from app.services.auth_service import hash_password, verify_password
from app.services.book_service import _db
from app.services.subscription_plans import PlanId, get_plan

Role = Literal["user", "admin"]
SubscriptionStatus = Literal["inactive", "active", "cancelled"]

_users_indexes_ensured = False


def _users() -> Collection:
    return _db()[MONGODB_USERS_COLLECTION]


def _ensure_indexes() -> None:
    global _users_indexes_ensured
    if _users_indexes_ensured:
        return
    col = _users()
    col.create_index([("email", ASCENDING)], unique=True)
    col.create_index([("user_id", ASCENDING)], unique=True)
    _users_indexes_ensured = True


def _serialize(doc: dict[str, Any]) -> dict[str, Any]:
    out = dict(doc)
    out.pop("_id", None)
    out.pop("password_hash", None)
    return out


def _public_user(doc: dict[str, Any]) -> dict[str, Any]:
    sub = doc.get("subscription") or {}
    return {
        "user_id": doc["user_id"],
        "email": doc["email"],
        "name": doc.get("name", ""),
        "role": doc.get("role", "user"),
        "subscription": {
            "plan_id": sub.get("plan_id"),
            "status": sub.get("status", "inactive"),
            "subscribed_at": sub.get("subscribed_at"),
        },
        "created_at": doc.get("created_at"),
    }


def create_user(
    *,
    email: str,
    password: str,
    name: str = "",
    role: Role = "user",
    plan_id: PlanId | None = None,
    subscription_status: SubscriptionStatus = "inactive",
) -> dict[str, Any]:
    _ensure_indexes()
    normalized = email.strip().lower()
    if _users().find_one({"email": normalized}):
        raise ValueError("An account with this email already exists.")

    now = int(time.time() * 1000)
    user_id = str(uuid.uuid4())
    subscription: dict[str, Any] = {
        "plan_id": plan_id,
        "status": subscription_status,
        "subscribed_at": now if subscription_status == "active" and plan_id else None,
    }
    doc = {
        "user_id": user_id,
        "email": normalized,
        "name": name.strip(),
        "password_hash": hash_password(password),
        "role": role,
        "subscription": subscription,
        "created_at": now,
    }
    _users().insert_one(doc)
    return _public_user(doc)


def get_user_by_email(email: str) -> dict[str, Any] | None:
    _ensure_indexes()
    doc = _users().find_one({"email": email.strip().lower()})
    return doc


def get_user_by_id(user_id: str) -> dict[str, Any] | None:
    _ensure_indexes()
    return _users().find_one({"user_id": user_id})


def authenticate_user(email: str, password: str) -> dict[str, Any] | None:
    doc = get_user_by_email(email)
    if not doc:
        return None
    if not verify_password(password, doc["password_hash"]):
        return None
    return doc


def subscribe_user(user_id: str, plan_id: PlanId) -> dict[str, Any] | None:
    if not get_plan(plan_id):
        raise ValueError(f"Unknown plan: {plan_id}")
    now = int(time.time() * 1000)
    result = _users().find_one_and_update(
        {"user_id": user_id},
        {
            "$set": {
                "subscription.plan_id": plan_id,
                "subscription.status": "active",
                "subscription.subscribed_at": now,
            }
        },
        return_document=True,
    )
    return _public_user(result) if result else None


def cancel_subscription(user_id: str) -> dict[str, Any] | None:
    result = _users().find_one_and_update(
        {"user_id": user_id},
        {"$set": {"subscription.status": "cancelled"}},
        return_document=True,
    )
    return _public_user(result) if result else None


def has_active_subscription(doc: dict[str, Any]) -> bool:
    sub = doc.get("subscription") or {}
    return sub.get("status") == "active" and bool(sub.get("plan_id"))


def user_plan_id(doc: dict[str, Any]) -> str | None:
    sub = doc.get("subscription") or {}
    if sub.get("status") != "active":
        return None
    plan_id = sub.get("plan_id")
    return str(plan_id) if plan_id else None


def count_user_books(owner_id: str) -> int:
    from app.services.book_service import count_books_for_owner

    return count_books_for_owner(owner_id)
