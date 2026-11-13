"""Wan TI2V chunk planning: ~5s single pass, longer videos as chained I2V segments."""

from __future__ import annotations

import math
from dataclasses import dataclass

# Official TI2V default (~5.04s at 24fps). Must be 4n+1.
DEFAULT_CHUNK_SECONDS = 5.0
DEFAULT_FPS = 24
DEFAULT_CHUNK_FRAMES = 121


@dataclass(frozen=True)
class WanChunkPlan:
    duration_seconds: float
    fps: int
    chunk_seconds: float
    frames_per_chunk: int
    num_chunks: int
    single_pass: bool
    chunk_actual_duration_seconds: float


def align_wan_frame_count(raw_frames: int) -> int:
    """Wan frame rule: 4n+1 (minimum 5). Round up to next valid count."""
    if raw_frames < 5:
        return 5
    n = math.ceil((raw_frames - 1) / 4)
    return int(n) * 4 + 1


def frames_for_duration(duration_seconds: float, fps: int) -> int:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    requested = max(1, int(round(duration_seconds * fps)))
    return align_wan_frame_count(requested)


def plan_wan_chunks(
    duration_seconds: float,
    *,
    fps: int = DEFAULT_FPS,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
) -> WanChunkPlan:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")

    frames_per_chunk = frames_for_duration(min(duration_seconds, chunk_seconds), fps)
    # Prefer the canonical 5s TI2V length when the request is at/above one chunk.
    if duration_seconds >= chunk_seconds:
        frames_per_chunk = max(frames_per_chunk, DEFAULT_CHUNK_FRAMES)
        # Keep 4n+1 if DEFAULT_CHUNK_FRAMES already is.
        frames_per_chunk = align_wan_frame_count(frames_per_chunk)

    chunk_actual = frames_per_chunk / float(fps)
    if duration_seconds <= chunk_seconds:
        return WanChunkPlan(
            duration_seconds=duration_seconds,
            fps=fps,
            chunk_seconds=chunk_seconds,
            frames_per_chunk=frames_per_chunk,
            num_chunks=1,
            single_pass=True,
            chunk_actual_duration_seconds=chunk_actual,
        )

    num_chunks = int(math.ceil(duration_seconds / chunk_seconds))
    return WanChunkPlan(
        duration_seconds=duration_seconds,
        fps=fps,
        chunk_seconds=chunk_seconds,
        frames_per_chunk=frames_per_chunk,
        num_chunks=num_chunks,
        single_pass=False,
        chunk_actual_duration_seconds=chunk_actual,
    )
