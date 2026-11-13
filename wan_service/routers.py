"""API router aggregation."""

from __future__ import annotations

from fastapi import APIRouter

from video_generation.router import router as video_generation_router
from wan_service.apps.health.api.routes import router as health_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(video_generation_router)
