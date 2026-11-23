"""Job submission and retrieval routes (multipart legacy + JSON video plan)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, cast

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import ValidationError

from core.types import VideoMode
from video_generation.schemas import VideoGenerateRequest
from video_generation.service import (
    finalize_job_enqueue,
    public_job_created_payload,
    submit_video_job,
    submit_video_job_from_request,
)
from wan_service import config as service_config
from wan_service.apps.jobs.services.manager import job_manager
from wan_service.apps.jobs.services.engine_dispatch import (
    evaluate_engine_readiness,
    normalize_engine,
)
from wan_service.utils.auth import require_worker_auth
from wan_service.utils.job_ids import require_job_id

router = APIRouter(prefix="/jobs", tags=["jobs"])

ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}
ALLOWED_VIDEO_MODES = {"text_to_video", "image_to_video"}


def _resolve_video_mode(
    *,
    engine: str,
    task: str | None,
    ltx_mode: str | None,
) -> VideoMode:
    raw = (task or ltx_mode or "").strip().lower()
    if not raw:
        return "text_to_video" if engine == "cogvideox" else "image_to_video"
    if raw not in ALLOWED_VIDEO_MODES:
        raise HTTPException(
            status_code=400,
            detail="task/mode must be text_to_video or image_to_video",
        )
    if engine == "cogvideox" and raw != "text_to_video":
        raise HTTPException(
            status_code=400,
            detail=(
                "CogVideoX POC supports text_to_video only. "
                "I2V requires a separate CogVideoX I2V checkpoint."
            ),
        )
    return cast(VideoMode, raw)


async def _create_job_multipart(
    request: Request,
    *,
    prompt: str,
    duration_seconds: float,
    engine_type: str,
    image: UploadFile | None,
    task: str | None,
    ltx_mode: str | None,
    negative_prompt: str | None,
    width: int | None,
    height: int | None,
    fps: int | None,
    seed: int | None,
) -> dict:
    if not prompt or not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")
    if len(prompt) > service_config.MAX_PROMPT_CHARS:
        raise HTTPException(status_code=400, detail="prompt too long")
    if duration_seconds <= 0 or duration_seconds > 60:
        raise HTTPException(status_code=400, detail="duration_seconds must be between 0 and 60")

    try:
        engine = normalize_engine(engine_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if negative_prompt and len(negative_prompt) > service_config.MAX_PROMPT_CHARS:
        raise HTTPException(status_code=400, detail="negative_prompt too long")

    mode = _resolve_video_mode(engine=engine, task=task, ltx_mode=ltx_mode)
    needs_image = mode == "image_to_video"
    if needs_image and image is None:
        raise HTTPException(status_code=400, detail="image is required for this engine/mode")
    if image is not None:
        content_type = (image.content_type or "").lower()
        if content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(status_code=415, detail="Unsupported image type")

    readiness = evaluate_engine_readiness(engine)
    if not readiness.ready:
        raise HTTPException(status_code=503, detail=readiness.public_reason)

    ctx = submit_video_job(
        mode=mode,
        prompt=prompt,
        duration_seconds=duration_seconds,
        engine_type=engine,
        image=image,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        fps=fps,
        seed=seed,
    )
    await finalize_job_enqueue(ctx, image=image, body=None)
    return public_job_created_payload(
        request=request,
        record=ctx["record"],
        mode=ctx["mode"],
        estimated_seconds=ctx["estimated_seconds"],
        queue_ahead=ctx["queue_ahead"],
    )


@router.post("")
async def create_job(
    request: Request,
    prompt: str | None = Form(None),
    duration_seconds: float | None = Form(None),
    engine_type: str | None = Form(None),
    image: Annotated[UploadFile | None, File()] = None,
    task: str | None = Form(None),
    ltx_mode: str | None = Form(None),
    negative_prompt: str | None = Form(None),
    width: int | None = Form(None),
    height: int | None = Form(None),
    fps: int | None = Form(None),
    seed: int | None = Form(None),
    _: None = Depends(require_worker_auth),
):
    """
    Create a generation job.

    - **JSON** (`application/json`): upstream shot plan (`VideoGenerateRequest`).
    - **Multipart**: legacy fields `prompt`, `duration_seconds`, optional `task` / `ltx_mode`.
    """
    content_type = (request.headers.get("content-type") or "").lower()

    if "application/json" in content_type:
        try:
            raw = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
        try:
            body = VideoGenerateRequest.model_validate(raw)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        ctx = submit_video_job_from_request(body, image=image)
        await finalize_job_enqueue(ctx, image=image, body=body)
        return public_job_created_payload(
            request=request,
            record=ctx["record"],
            mode=ctx["mode"],
            estimated_seconds=ctx["estimated_seconds"],
            queue_ahead=ctx["queue_ahead"],
        )

    if prompt is None:
        raise HTTPException(
            status_code=400,
            detail="Send application/json (shot plan) or multipart with prompt",
        )

    return await _create_job_multipart(
        request,
        prompt=prompt,
        duration_seconds=float(duration_seconds if duration_seconds is not None else 5.0),
        engine_type=engine_type or "wan",
        image=image,
        task=task,
        ltx_mode=ltx_mode,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        fps=fps,
        seed=seed,
    )


def _job_not_found(job_id: str) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail={
            "message": "Job not found",
            "job_id": job_id,
            "revoke_polling": True,
        },
    )


@router.get("/{job_id}/progress")
def get_job_progress(job_id: str, request: Request):
    """Poll job status with only the job id (Bearer or ?api_key=)."""
    job_id = require_job_id(job_id)
    require_worker_auth(request)
    record = job_manager.get(job_id)
    if not record:
        raise _job_not_found(job_id)
    return record.to_progress_dict(base_url=str(request.base_url).rstrip("/"))


@router.get("/{job_id}")
def get_job(job_id: str, request: Request):
    job_id = require_job_id(job_id)
    require_worker_auth(request)
    record = job_manager.get(job_id)
    if not record:
        raise _job_not_found(job_id)
    return record.to_dict(base_url=str(request.base_url).rstrip("/"))


@router.get("/{job_id}/video")
def get_job_video(job_id: str, request: Request):
    job_id = require_job_id(job_id)
    require_worker_auth(request)
    record = job_manager.get(job_id)
    if not record:
        raise _job_not_found(job_id)
    if record.status != "completed" or not record.video_path:
        raise HTTPException(status_code=404, detail="Video not ready")
    path = Path(record.video_path).resolve()
    work_root = record.work_dir.resolve() if record.work_dir else None
    if work_root and work_root not in path.parents:
        raise HTTPException(status_code=400, detail="Invalid video path")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video file missing")
    return FileResponse(path, media_type="video/mp4", filename=f"{job_id}.mp4")