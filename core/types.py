"""Shared job parameters for all animation engines."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

EngineType = Literal["wan", "ltx", "cogvideox"]
VideoMode = Literal["text_to_video", "image_to_video"]
LtxMode = VideoMode  # backward-compatible alias

LogCallback = Callable[[str], None]


@dataclass
class GenerationJobParams:
    job_id: str
    prompt: str
    output_mp4: Path
    image_path: Path | None = None
    duration_seconds: float | None = None
    negative_prompt: str | None = None
    width: int | None = None
    height: int | None = None
    fps: int = 24
    mode: VideoMode | None = None
    seed: int | None = None
    on_log_line: LogCallback | None = None
    metadata: dict = field(default_factory=dict)
