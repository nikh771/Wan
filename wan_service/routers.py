"""API router aggregation."""

from __future__ import annotations

from fastapi import APIRouter

from wan_service.apps.health.api.routes import router as health_router
from wan_service.apps.jobs.api.routes import router as jobs_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(jobs_router)
