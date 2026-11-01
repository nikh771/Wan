"""Bearer token authentication for the Wan service."""

from __future__ import annotations

import secrets

from fastapi import HTTPException, Request

from wan_service.core.config import settings


def worker_auth_token(request: Request) -> str:
    auth = request.headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return (request.query_params.get("api_key") or "").strip()


def require_worker_auth(request: Request) -> None:
    key = settings.WAN_WORKER_API_KEY
    if not key:
        raise HTTPException(
            status_code=503,
            detail="WAN_WORKER_API_KEY is not configured on the worker.",
        )
    token = worker_auth_token(request)
    if not token or not secrets.compare_digest(token, key):
        detail: str | dict[str, object] = "Unauthorized"
        if request.method == "GET" and request.url.path.startswith("/jobs/"):
            detail = {
                "message": "Unauthorized — set Bearer token or ?api_key=",
                "revoke_polling": True,
            }
        raise HTTPException(status_code=401, detail=detail)
