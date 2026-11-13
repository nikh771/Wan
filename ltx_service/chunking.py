"""LTX duration chunk planning: ≤10s single pass; longer jobs as chained I2V segments."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ltx_service.config import align_ltx_frame_count

DEFAULT_CHUNK_SECONDS = 10.0
DEFAULT_FPS = 24


@dataclass(frozen=True)
class LtxChunkPlan:
    duration_seconds: float
    fps: int
    chunk_seconds: float
    frames_per_chunk: int
    num_chunks: int
    single_pass: bool
    chunk_actual_duration_seconds: float
    last_chunk_duration_seconds: float
    last_chunk_frames: int


def frames_for_duration(duration_seconds: float, fps: int) -> int:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    requested = max(1, int(round(duration_seconds * fps)))
    return align_ltx_frame_count(requested)


def plan_ltx_chunks(
    duration_seconds: float,
    *,
    fps: int = DEFAULT_FPS,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
) -> LtxChunkPlan:
    """Split a request into ≤chunk_seconds segments (LTX frame rule 8N+1)."""
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")

    if duration_seconds <= chunk_seconds:
        frames = frames_for_duration(duration_seconds, fps)
        actual = frames / float(fps)
        return LtxChunkPlan(
            duration_seconds=duration_seconds,
            fps=fps,
            chunk_seconds=chunk_seconds,
            frames_per_chunk=frames,
            num_chunks=1,
            single_pass=True,
            chunk_actual_duration_seconds=actual,
            last_chunk_duration_seconds=actual,
            last_chunk_frames=frames,
        )

    num_chunks = int(math.ceil(duration_seconds / chunk_seconds))
    full_frames = frames_for_duration(chunk_seconds, fps)
    full_actual = full_frames / float(fps)

    # Last chunk covers the remainder (at least one LTX-valid frame count).
    remainder = duration_seconds - (num_chunks - 1) * chunk_seconds
    remainder = max(remainder, 1.0 / fps)
    last_frames = frames_for_duration(remainder, fps)
    last_actual = last_frames / float(fps)

    return LtxChunkPlan(
        duration_seconds=duration_seconds,
        fps=fps,
        chunk_seconds=chunk_seconds,
        frames_per_chunk=full_frames,
        num_chunks=num_chunks,
        single_pass=False,
        chunk_actual_duration_seconds=full_actual,
        last_chunk_duration_seconds=last_actual,
        last_chunk_frames=last_frames,
    )


def chunk_duration_seconds(plan: LtxChunkPlan, index: int) -> float:
    if index < 0 or index >= plan.num_chunks:
        raise IndexError(f"chunk index {index} out of range for {plan.num_chunks} chunks")
    if index == plan.num_chunks - 1:
        return plan.last_chunk_duration_seconds
    return plan.chunk_actual_duration_seconds


def chunk_frame_count(plan: LtxChunkPlan, index: int) -> int:
    if index < 0 or index >= plan.num_chunks:
        raise IndexError(f"chunk index {index} out of range for {plan.num_chunks} chunks")
    if index == plan.num_chunks - 1:
        return plan.last_chunk_frames
    return plan.frames_per_chunk
