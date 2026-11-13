"""LTX HTTP routes (mounted by wan_service API)."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ltx_service.health import ltx_health_payload
from wan_service.utils.auth import require_worker_auth

router = APIRouter(prefix="/video", tags=["video", "ltx"])

api_v1_router = APIRouter(prefix="/api/v1/video", tags=["video-v1", "ltx"])


@router.get("/engines/ltx/health")
def ltx_engine_health(request: Request) -> dict:
    require_worker_auth(request)
    return ltx_health_payload()


api_v1_router.add_api_route(
    "/engines/ltx/health",
    ltx_engine_health,
    methods=["GET"],
)
