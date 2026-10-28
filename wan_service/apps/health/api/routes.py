"""Health routes."""

from __future__ import annotations

from fastapi import APIRouter

from wan_service.core.runtime import health_payload

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    return health_payload()
