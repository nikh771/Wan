"""LTX job facade for the shared FastAPI job queue."""

from __future__ import annotations

from ltx_service.runner import run_ltx_generation
from core.types import GenerationJobParams


def run_ltx_engine(params: GenerationJobParams) -> dict:
    return run_ltx_generation(params)
