"""Engine dispatch — single import surface for video_generation."""

from core.engine_dispatch import (
    assert_engine_available,
    evaluate_engine_readiness,
    normalize_engine,
    run_engine,
)

__all__ = [
    "assert_engine_available",
    "evaluate_engine_readiness",
    "normalize_engine",
    "run_engine",
]
