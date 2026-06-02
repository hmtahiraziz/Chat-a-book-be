"""Subscription plan definitions (no payment gateway)."""

from __future__ import annotations

from typing import Any, Literal

PlanId = Literal["starter", "pro"]

PLANS: dict[PlanId, dict[str, Any]] = {
    "starter": {
        "id": "starter",
        "name": "Starter",
        "price_monthly": 9.99,
        "currency": "USD",
        "description": "For individual readers getting started with book chat.",
        "features": [
            "Up to 5 books in your library",
            "RAG chat with source citations",
            "Chat session history",
            "Standard response speed",
        ],
        "limits": {
            "max_books": 5,
            "tts_enabled": False,
        },
    },
    "pro": {
        "id": "pro",
        "name": "Pro",
        "price_monthly": 24.99,
        "currency": "USD",
        "description": "For power users who need more capacity and voice output.",
        "features": [
            "Unlimited books in your library",
            "RAG chat with source citations",
            "Chat session history",
            "Text-to-speech for replies",
            "Priority indexing queue",
        ],
        "limits": {
            "max_books": None,
            "tts_enabled": True,
        },
    },
}


def list_plans_public() -> list[dict[str, Any]]:
    return [
        {
            "id": p["id"],
            "name": p["name"],
            "price_monthly": p["price_monthly"],
            "currency": p["currency"],
            "description": p["description"],
            "features": p["features"],
        }
        for p in PLANS.values()
    ]


def get_plan(plan_id: str) -> dict[str, Any] | None:
    return PLANS.get(plan_id)  # type: ignore[arg-type]


def plan_limits(plan_id: str | None) -> dict[str, Any]:
    if not plan_id or plan_id not in PLANS:
        return {"max_books": 0, "tts_enabled": False}
    return dict(PLANS[plan_id]["limits"])  # type: ignore[index]
