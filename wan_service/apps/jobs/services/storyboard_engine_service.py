"""Storyboard engine — CPU camera motion (no diffusion)."""

from __future__ import annotations

from wan_service.apps.jobs.storyboard_renderer import render_storyboard_mp4
from wan_service.engines.types import GenerationJobParams


def run_storyboard_engine(params: GenerationJobParams) -> dict:
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
