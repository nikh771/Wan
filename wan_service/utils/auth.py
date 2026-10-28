"""Bearer token authentication for the Wan service."""

from __future__ import annotations

import secrets

from fastapi import HTTPException, Request

from wan_service.core.config import settings


def require_worker_auth(request: Request) -> None:
    key = settings.WAN_WORKER_API_KEY
    if not key:
        raise HTTPException(
            status_code=503,
            detail="WAN_WORKER_API_KEY is not configured on the worker.",
        )
    auth = request.headers.get("Authorization") or ""
    token = ""
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
    if not token or not secrets.compare_digest(token, key):
        raise HTTPException(status_code=401, detail="Unauthorized")
