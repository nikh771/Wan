"""API router aggregation."""

from __future__ import annotations

from fastapi import APIRouter

from cogvideox_service.api.routes import (
    api_v1_router as cogvideox_api_v1_router,
    router as cogvideox_api_router,
)
from ltx_service.api.routes import api_v1_router as ltx_api_v1_router, router as ltx_api_router
from video_generation.router import router as video_generation_router
from wan_service.apps.health.api.routes import router as health_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(video_generation_router)
api_router.include_router(ltx_api_router)
api_router.include_router(cogvideox_api_router)
api_router.include_router(ltx_api_v1_router)
api_router.include_router(cogvideox_api_v1_router)
