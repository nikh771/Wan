"""Wan TI2V job facade (delegates to existing inference subprocess)."""

from __future__ import annotations

from core.types import GenerationJobParams
from wan_service.apps.jobs.services.inference import run_wan_inference


def run_wan_engine(params: GenerationJobParams) -> None:
    if params.image_path is None or not params.image_path.is_file():
        raise FileNotFoundError("Wan image-to-video requires an input image.")
    run_wan_inference(
        job_id=params.job_id,
        image_path=params.image_path,
        prompt=params.prompt,
        output_mp4=params.output_mp4,
        on_log_line=params.on_log_line,
    )
