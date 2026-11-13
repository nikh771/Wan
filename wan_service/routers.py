"""API router aggregation."""

from __future__ import annotations

from fastapi import APIRouter

from wan_service.apps.health.api.routes import router as health_router
from wan_service.apps.jobs.api.routes import router as jobs_router
from ltx_service.api.routes import api_v1_router, router as ltx_video_router
from cogvideox_service.api.routes import (
    api_v1_router as cogvideox_api_v1_router,
    router as cogvideox_video_router,
)

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(jobs_router)
api_router.include_router(ltx_video_router)
api_router.include_router(api_v1_router)
api_router.include_router(cogvideox_video_router)
api_router.include_router(cogvideox_api_v1_router)
