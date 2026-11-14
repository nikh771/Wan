"""Job creation orchestration for the unified video API."""

from __future__ import annotations

import shutil

from fastapi import HTTPException, UploadFile

from core.types import EngineType, VideoMode
from video_generation.engines import evaluate_engine_readiness, normalize_engine
from video_generation.estimates import estimate_processing_seconds
from video_generation.modes import normalize_video_mode
from video_generation.schemas import VideoGenerateRequest
from wan_service import config as service_config
from wan_service.apps.jobs.services.manager import job_manager


def _queue_depth() -> int:
    with job_manager._lock:
        active = 1 if job_manager._active else 0
        return len(job_manager._queue) + active


def submit_video_job(
    *,
    mode: str | VideoMode,
    prompt: str,
    duration_seconds: float,
    engine_type: str,
    image: UploadFile | None,
    negative_prompt: str | None,
    width: int | None,
    height: int | None,
    fps: int | None,
    seed: int | None,
    request_metadata: dict | None = None,
    image_pending: bool = False,
) -> dict:
    if isinstance(mode, str):
        try:
            video_mode = normalize_video_mode(mode)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        video_mode = mode

    if not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")
    if len(prompt) > service_config.MAX_PROMPT_CHARS:
        raise HTTPException(status_code=400, detail="prompt too long")
    if duration_seconds <= 0 or duration_seconds > 60:
        raise HTTPException(status_code=400, detail="duration_seconds must be between 0 and 60")

    try:
        engine: EngineType = normalize_engine(engine_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if video_mode == "image_to_video" and image is None and not image_pending:
        raise HTTPException(
            status_code=400,
            detail="image is required for i2v (multipart file, image_url, or image_base64)",
        )
    if engine == "cogvideox" and video_mode != "text_to_video":
        raise HTTPException(
            status_code=400,
            detail="CogVideoX POC supports text_to_video only.",
        )

    readiness = evaluate_engine_readiness(engine)
    if not readiness.ready:
        raise HTTPException(status_code=503, detail=readiness.public_reason)

    if engine == "cogvideox":
        width = 720 if width is None else width
        height = 480 if height is None else height
        fps = 8 if fps is None else fps

    queue_ahead = _queue_depth()
    estimated_seconds = estimate_processing_seconds(
        engine=engine,
        duration_seconds=duration_seconds,
        queue_ahead=queue_ahead,
    )

    try:
        record = job_manager.create_job(
            prompt=prompt.strip(),
            duration_seconds=duration_seconds,
            engine_type=engine,
            negative_prompt=negative_prompt.strip() if negative_prompt else None,
            width=width,
            height=height,
            fps=fps,
            ltx_mode=video_mode,
            seed=seed,
            estimated_total_seconds=estimated_seconds,
            request_metadata=request_metadata,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    work_dir = record.work_dir
    if work_dir is None:
        raise HTTPException(status_code=503, detail="Job work directory was not created")

    if image is not None:
        content_type = (image.content_type or "").lower()
        allowed = {"image/png", "image/jpeg", "image/webp"}
        if content_type not in allowed:
            raise HTTPException(status_code=415, detail="Unsupported image type")

    return {
        "record": record,
        "mode": video_mode,
        "engine": engine,
        "estimated_seconds": estimated_seconds,
        "queue_ahead": queue_ahead,
        "image": image,
        "content_type": (image.content_type or "").lower() if image else None,
        "has_inline_image": image is not None,
    }


def submit_video_job_from_request(
    body: VideoGenerateRequest,
    *,
    image: UploadFile | None = None,
) -> dict:
    """JSON shot payload → queued job (primary upstream integration)."""
    try:
        video_mode = normalize_video_mode(body.mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    image_pending = video_mode == "image_to_video" and (
        image is not None or bool(body.image_url) or bool(body.image_base64)
    )
    return submit_video_job(
        mode=body.mode,
        prompt=body.prompt,
        duration_seconds=body.duration_seconds,
        engine_type=body.engine_type,
        image=image,
        negative_prompt=body.negative_prompt,
        width=body.width,
        height=body.height,
        fps=body.fps,
        seed=body.seed,
        request_metadata=body.correlation_metadata(),
        image_pending=image_pending,
    )


async def persist_image_and_enqueue(job_id: str, image: UploadFile, content_type: str) -> None:
    record = job_manager.get(job_id)
    if not record or not record.work_dir:
        raise HTTPException(status_code=500, detail="Job missing work directory")
    data = await image.read()
    if len(data) > service_config.MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image too large")
    if len(data) < 100:
        raise HTTPException(status_code=400, detail="Empty image")
    suffix = ".png" if "png" in content_type else ".jpg" if "jpeg" in content_type else ".webp"
    input_path = record.work_dir / f"input{suffix}"
    input_path.write_bytes(data)
    target = record.work_dir / "input.png"
    if input_path != target:
        shutil.copy2(input_path, target)
    job_manager.enqueue(job_id)


async def persist_image_bytes_and_enqueue(
    job_id: str, data: bytes, content_type: str
) -> None:
    record = job_manager.get(job_id)
    if not record or not record.work_dir:
        raise HTTPException(status_code=500, detail="Job missing work directory")
    if len(data) > service_config.MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image too large")
    from video_generation.images import write_job_input_image

    write_job_input_image(record.work_dir, data, content_type)
    job_manager.enqueue(job_id)