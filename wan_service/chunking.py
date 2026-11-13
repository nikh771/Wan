"""Wan TI2V chunk planning: ~10s single pass, longer videos as chained I2V segments."""

from __future__ import annotations

import math
from dataclasses import dataclass

# ~10s at 24fps (4n+1). Chained via last-frame I2V for continuity.
DEFAULT_CHUNK_SECONDS = 10.0
DEFAULT_FPS = 24


@dataclass(frozen=True)
class WanChunkPlan:
    duration_seconds: float
    fps: int
    chunk_seconds: float
    frames_per_chunk: int
    num_chunks: int
    single_pass: bool
    chunk_actual_duration_seconds: float
    last_chunk_duration_seconds: float
    last_chunk_frames: int


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
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")

    if duration_seconds <= chunk_seconds:
        frames = frames_for_duration(duration_seconds, fps)
        actual = frames / float(fps)
        return WanChunkPlan(
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
    remainder = duration_seconds - (num_chunks - 1) * chunk_seconds
    remainder = max(remainder, 1.0 / fps)
    last_frames = frames_for_duration(remainder, fps)
    last_actual = last_frames / float(fps)

    return WanChunkPlan(
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


def chunk_duration_seconds(plan: WanChunkPlan, index: int) -> float:
    if index == plan.num_chunks - 1:
        return plan.last_chunk_duration_seconds
    return plan.chunk_actual_duration_seconds


def chunk_frame_count(plan: WanChunkPlan, index: int) -> int:
    if index == plan.num_chunks - 1:
        return plan.last_chunk_frames
    return plan.frames_per_chunk
