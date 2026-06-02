"""Registration, login, and subscription routes."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from app.deps.auth import get_current_user
from app.models import (
    DemoLoginResponse,
    LoginRequest,
    RegisterRequest,
    SubscribeRequest,
    TokenResponse,
    UserPublic,
)
from app.services.auth_service import create_access_token
from app.services.demo_service import ensure_demo_user, resolve_demo_book
from app.services.subscription_plans import PlanId, list_plans_public
from app.services.user_service import (
    authenticate_user,
    cancel_subscription,
    create_user,
    subscribe_user,
)

router = APIRouter(prefix="/auth", tags=["auth"])


def _to_user_public(doc: dict[str, Any]) -> UserPublic:
    sub = doc.get("subscription") or {}
    return UserPublic(
        user_id=doc["user_id"],
        email=doc["email"],
        name=doc.get("name", ""),
        role=doc.get("role", "user"),
        subscription={
            "plan_id": sub.get("plan_id"),
            "status": sub.get("status", "inactive"),
            "subscribed_at": sub.get("subscribed_at"),
        },
        created_at=doc.get("created_at"),
    )


@router.get("/plans")
def list_subscription_plans() -> dict[str, Any]:
    return {"plans": list_plans_public()}


@router.get("/demo")
def demo_info() -> dict[str, Any]:
    """Public metadata for the landing-page demo (no auth)."""
    book_id, label = resolve_demo_book()
    return {
        "demo_book_id": book_id,
        "demo_book_label": label,
        "demo_login_route": "POST /auth/demo-login",
    }


@router.post("/demo-login", response_model=DemoLoginResponse)
def demo_login() -> DemoLoginResponse:
    """Sign in as the shared demo user — no password or registration."""
    try:
        doc = ensure_demo_user()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    from app.services.user_service import _public_user

    public = _public_user(doc)
    book_id, label = resolve_demo_book()
    token = create_access_token(
        user_id=public["user_id"],
        email=public["email"],
        role=public["role"],
    )
    return DemoLoginResponse(
        access_token=token,
        user=_to_user_public(public),
        demo_book_id=book_id,
        demo_book_label=label,
    )


@router.post("/register", response_model=TokenResponse, status_code=201)
def register(body: RegisterRequest) -> TokenResponse:
    try:
        doc = create_user(
            email=body.email,
            password=body.password,
            name=body.name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    token = create_access_token(
        user_id=doc["user_id"],
        email=doc["email"],
        role=doc["role"],
    )
    return TokenResponse(access_token=token, user=_to_user_public(doc))


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest) -> TokenResponse:
    doc = authenticate_user(body.email, body.password)
    if not doc:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    token = create_access_token(
        user_id=doc["user_id"],
        email=doc["email"],
        role=doc["role"],
    )
    from app.services.user_service import _public_user

    return TokenResponse(access_token=token, user=_to_user_public(_public_user(doc)))


@router.get("/me", response_model=UserPublic)
def me(user: Annotated[dict[str, Any], Depends(get_current_user)]) -> UserPublic:
    from app.services.user_service import _public_user

    return _to_user_public(_public_user(user))


@router.post("/subscribe", response_model=UserPublic)
def subscribe(
    body: SubscribeRequest,
    user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> UserPublic:
    """Mock subscription — activates plan without payment."""
    try:
        updated = subscribe_user(user["user_id"], body.plan_id)  # type: ignore[arg-type]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not updated:
        raise HTTPException(status_code=404, detail="User not found.")
    return _to_user_public(updated)


@router.post("/subscribe/mock-checkout", response_model=UserPublic)
def mock_checkout(
    body: SubscribeRequest,
    user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> UserPublic:
    """Alias for mock payment UI — same as /auth/subscribe."""
    try:
        updated = subscribe_user(user["user_id"], body.plan_id)  # type: ignore[arg-type]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not updated:
        raise HTTPException(status_code=404, detail="User not found.")
    return _to_user_public(updated)


@router.post("/cancel-subscription", response_model=UserPublic)
def cancel_sub(user: Annotated[dict[str, Any], Depends(get_current_user)]) -> UserPublic:
    updated = cancel_subscription(user["user_id"])
    if not updated:
        raise HTTPException(status_code=404, detail="User not found.")
    return _to_user_public(updated)


@router.get("/subscription/status")
def subscription_status(
    user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> dict[str, Any]:
    from app.services.user_service import has_active_subscription, user_plan_id

    plan_id = user_plan_id(user)
    return {
        "active": has_active_subscription(user),
        "plan_id": plan_id,
        "role": user.get("role", "user"),
    }
