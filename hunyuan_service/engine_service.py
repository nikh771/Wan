"""Hunyuan job facade for the shared FastAPI job queue."""

from __future__ import annotations

from core.types import GenerationJobParams
from hunyuan_service.runner import run_hunyuan_generation


def run_hunyuan_engine(params: GenerationJobParams) -> dict:
    return run_hunyuan_generation(params)