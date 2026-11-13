"""Wan TI2V job facade: single-pass ≤5s, chunked longer videos with last-frame continuity."""

from __future__ import annotations

import logging
from pathlib import Path

from core.types import GenerationJobParams
from wan_service.apps.jobs.services.inference import run_wan_inference, select_wan_size
from wan_service.chunking import DEFAULT_CHUNK_SECONDS, plan_wan_chunks
from wan_service.utils.ffmpeg import concat_mp4s, extract_last_frame

logger = logging.getLogger(__name__)


def run_wan_engine(params: GenerationJobParams) -> dict:
    mode = (params.mode or "image_to_video").strip().lower()
    if mode not in ("text_to_video", "image_to_video"):
        raise ValueError("Wan mode must be text_to_video or image_to_video")

    if mode == "image_to_video":
        if params.image_path is None or not params.image_path.is_file():
            raise FileNotFoundError("Wan image_to_video requires an input image.")

    duration = float(params.duration_seconds) if params.duration_seconds else DEFAULT_CHUNK_SECONDS
    fps = int(params.fps or 24)
    plan = plan_wan_chunks(duration, fps=fps, chunk_seconds=DEFAULT_CHUNK_SECONDS)
    size = select_wan_size(params.image_path, width=params.width, height=params.height)
    work_dir = params.output_mp4.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        logger.info("%s", msg)
        if params.on_log_line is not None:
            params.on_log_line(msg)

    if plan.single_pass:
        _log(
            f"[wan_service] Single-pass {mode} | frames={plan.frames_per_chunk} "
            f"duration≈{plan.chunk_actual_duration_seconds:.2f}s"
        )
        run_wan_inference(
            job_id=params.job_id,
            image_path=params.image_path if mode == "image_to_video" else None,
            prompt=params.prompt,
            output_mp4=params.output_mp4,
            on_log_line=params.on_log_line,
            frame_num=plan.frames_per_chunk,
            seed=params.seed,
            width=params.width,
            height=params.height,
            size=size,
        )
        return {
            "engine": "wan",
            "engine_used": "wan",
            "mode": mode,
            "chunks": 1,
            "frames_per_chunk": plan.frames_per_chunk,
            "duration_seconds": duration,
        }

    _log(
        f"[wan_service] Chunked {mode} | chunks={plan.num_chunks} "
        f"× ~{plan.chunk_actual_duration_seconds:.2f}s "
        f"(target {duration:.2f}s, last-frame continuity)"
    )
    chunk_paths: list[Path] = []
    bridge: Path | None = params.image_path if mode == "image_to_video" else None

    for i in range(plan.num_chunks):
        chunk_out = work_dir / f"chunk_{i:02d}.mp4"
        # First chunk: T2V has no image; I2V uses user image.
        # Later chunks: always I2V from previous last frame.
        chunk_image = bridge
        chunk_seed = None if params.seed is None else int(params.seed) + i
        _log(
            f"[wan_service] Generating chunk {i + 1}/{plan.num_chunks} "
            f"({'i2v' if chunk_image is not None else 't2v'})"
        )
        run_wan_inference(
            job_id=f"{params.job_id}_c{i}",
            image_path=chunk_image,
            prompt=params.prompt,
            output_mp4=chunk_out,
            on_log_line=params.on_log_line,
            frame_num=plan.frames_per_chunk,
            seed=chunk_seed,
            width=params.width,
            height=params.height,
            size=size,
        )
        chunk_paths.append(chunk_out)
        if i < plan.num_chunks - 1:
            bridge = work_dir / f"bridge_{i + 1:02d}.png"
            extract_last_frame(chunk_out, bridge)
            _log(f"[wan_service] Extracted bridge frame for chunk {i + 2}: {bridge.name}")

    concat_mp4s(
        chunk_paths,
        params.output_mp4,
        skip_first_frame_after_first=True,
        fps=fps,
    )

    return {
        "engine": "wan",
        "engine_used": "wan",
        "mode": mode,
        "chunks": plan.num_chunks,
        "frames_per_chunk": plan.frames_per_chunk,
        "duration_seconds": duration,
        "chunk_files": [p.name for p in chunk_paths],
    }
