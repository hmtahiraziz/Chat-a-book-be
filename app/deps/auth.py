"""FastAPI authentication dependencies."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, Header, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import ADMIN_API_TOKEN
from app.services.auth_service import decode_access_token
from app.services.subscription_plans import plan_limits
from app.services.user_service import get_user_by_id, has_active_subscription, user_plan_id

_bearer = HTTPBearer(auto_error=False)


async def get_current_user_optional(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> dict[str, Any] | None:
    if credentials is None or credentials.scheme.lower() != "bearer":
        return None
    try:
        payload = decode_access_token(credentials.credentials)
    except ValueError:
        return None
    user_id = payload.get("sub")
    if not user_id:
        return None
    doc = get_user_by_id(str(user_id))
    if not doc:
        return None
    return doc


async def get_current_user(
    user: Annotated[dict[str, Any] | None, Depends(get_current_user_optional)],
) -> dict[str, Any]:
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Sign in and include Authorization: Bearer <token>.",
        )
    return user


async def require_active_subscription(
    user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> dict[str, Any]:
    if user.get("role") == "admin":
        return user
    if not has_active_subscription(user):
        raise HTTPException(
            status_code=403,
            detail="An active subscription is required. Choose a plan at POST /auth/subscribe.",
        )
    return user


async def require_admin_user(
    user: Annotated[dict[str, Any], Depends(get_current_user)],
    x_admin_token: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    if user.get("role") == "admin":
        return user
    if ADMIN_API_TOKEN and x_admin_token == ADMIN_API_TOKEN:
        return user
    raise HTTPException(status_code=403, detail="Admin access required.")


def enforce_book_limit(user: dict[str, Any]) -> None:
    if user.get("role") == "admin":
        return
    plan_id = user_plan_id(user)
    limits = plan_limits(plan_id)
    max_books = limits.get("max_books")
    if max_books is None:
        return
    from app.services.user_service import count_user_books

    owner_id = user["user_id"]
    if count_user_books(owner_id) >= int(max_books):
        raise HTTPException(
            status_code=403,
            detail=f"Your plan allows up to {max_books} books. Upgrade to Pro for unlimited uploads.",
        )


def enforce_tts_allowed(user: dict[str, Any]) -> None:
    if user.get("role") == "admin":
        return
    plan_id = user_plan_id(user)
    limits = plan_limits(plan_id)
    if not limits.get("tts_enabled"):
        raise HTTPException(
            status_code=403,
            detail="Text-to-speech is available on the Pro plan. Upgrade to enable voice output.",
        )
