"""Wan TI2V engine — delegates to existing inference (unchanged behaviour)."""

from __future__ import annotations

from wan_service.apps.jobs.services.inference import run_wan_inference
from wan_service.engines.types import GenerationJobParams


def run_wan_engine(params: GenerationJobParams) -> None:
    run_wan_inference(
        job_id=params.job_id,
        image_path=params.image_path,
        prompt=params.prompt,
        output_mp4=params.output_mp4,
        on_log_line=params.on_log_line,
    )
