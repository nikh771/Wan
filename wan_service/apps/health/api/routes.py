"""Health routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from wan_service.core.runtime import engine_health_payload

router = APIRouter(tags=["health"])


@router.get("/health")
def health(
    engine_type: str = Query(
        ...,
        description="wan | ltx | cogvideox",
    ),
) -> dict:
    try:
        return engine_health_payload(engine_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
