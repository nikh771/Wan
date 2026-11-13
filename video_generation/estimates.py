"""Rough wall-clock estimates for caller-facing ETA."""

from __future__ import annotations

from core.types import EngineType

# Seconds per ~10s chunk of output at 720p-class workloads (heuristic, not a guarantee).
_ENGINE_BASE_SEC: dict[EngineType, float] = {
    "wan": 50.0,
    "ltx": 95.0,
    "cogvideox": 130.0,
}

_LOAD_OVERHEAD_SEC: dict[EngineType, float] = {
    "wan": 25.0,
    "ltx": 35.0,
    "cogvideox": 40.0,
}

_POSTPROCESS_SEC = 6.0
_QUEUE_SLOT_SEC = 90.0


def estimate_processing_seconds(
    *,
    engine: EngineType,
    duration_seconds: float,
    queue_ahead: int = 0,
) -> float:
    """Total seconds until result is likely ready (queued + run + post)."""
    duration = max(0.5, float(duration_seconds))
    chunks = max(1, int((duration + 9.99) // 10.0))
    per_chunk = _ENGINE_BASE_SEC.get(engine, 60.0)
    load = _LOAD_OVERHEAD_SEC.get(engine, 20.0)
    run = load + chunks * per_chunk
    queue = max(0, queue_ahead) * _QUEUE_SLOT_SEC
    return queue + run + _POSTPROCESS_SEC


def estimate_remaining_seconds(
    *,
    estimated_total_seconds: float,
    progress_percent: int | None,
    elapsed_seconds: float,
) -> float:
    if progress_percent is not None and progress_percent > 0:
        total = max(estimated_total_seconds, elapsed_seconds)
        remaining = total * (1.0 - min(progress_percent, 100) / 100.0)
        return max(0.0, remaining)
    return max(0.0, estimated_total_seconds - elapsed_seconds)
