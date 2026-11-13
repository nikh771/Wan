"""CogVideoX job facade for the shared FastAPI job queue."""

from __future__ import annotations

from cogvideox_service.runner import run_cogvideox_generation
from core.types import GenerationJobParams


def run_cogvideox_engine(params: GenerationJobParams) -> dict:
    return run_cogvideox_generation(params)
