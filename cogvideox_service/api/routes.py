"""CogVideoX HTTP routes (mounted by wan_service API)."""

from __future__ import annotations

from fastapi import APIRouter, Request

from cogvideox_service.health import cogvideox_health_payload
from wan_service.utils.auth import require_worker_auth

router = APIRouter(prefix="/video", tags=["video", "cogvideox"])

api_v1_router = APIRouter(prefix="/api/v1/video", tags=["video-v1", "cogvideox"])


@router.get("/engines/cogvideox/health")
def cogvideox_engine_health(request: Request) -> dict:
    require_worker_auth(request)
    return cogvideox_health_payload()


api_v1_router.add_api_route(
    "/engines/cogvideox/health",
    cogvideox_engine_health,
    methods=["GET"],
)
