"""Storyboard job facade."""

from __future__ import annotations

from core.types import GenerationJobParams
from storyboard_service.renderer import render_storyboard_mp4


def run_storyboard_engine(params: GenerationJobParams) -> dict:
    if params.image_path is None or not params.image_path.is_file():
        raise FileNotFoundError("Storyboard requires an input image.")
    duration = float(params.duration_seconds or 5.0)
    return render_storyboard_mp4(
        image_path=params.image_path,
        output_path=params.output_mp4,
        duration_seconds=duration,
        width=params.width,
        height=params.height,
        fps=params.fps,
        on_log=params.on_log_line,
    )
