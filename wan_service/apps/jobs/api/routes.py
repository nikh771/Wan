"""Job submission and retrieval routes."""

from __future__ import annotations

import shutil
from pathlib import Path

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

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
) -> str | None:
    raw = (task or ltx_mode or "").strip().lower()
    if engine == "storyboard":
        return None
    if not raw:
        # CogVideoX POC is T2V-only; default other engines to I2V for backward compat.
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
    return raw


@router.post("")
async def create_job(
    request: Request,
    prompt: str = Form(...),
    duration_seconds: float = Form(5.0),
    engine_type: str = Form("wan"),
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
    needs_image = engine == "storyboard" or mode == "image_to_video"
    if needs_image and image is None:
        raise HTTPException(status_code=400, detail="image is required for this engine/mode")
    if image is not None:
        content_type = (image.content_type or "").lower()
        if content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(status_code=415, detail="Unsupported image type")

    readiness = evaluate_engine_readiness(engine)
    if not readiness.ready:
        raise HTTPException(status_code=503, detail=readiness.public_reason)

    data = b""
    if image is not None:
        data = await image.read()
        if len(data) > service_config.MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Image too large")
        if len(data) < 100:
            raise HTTPException(status_code=400, detail="Empty image")

    # CogVideoX official defaults when clients omit geometry/fps.
    if engine == "cogvideox":
        width = 720 if width is None else width
        height = 480 if height is None else height
        fps = 8 if fps is None else fps

    try:
        record = job_manager.create_job(
            prompt=prompt.strip(),
            duration_seconds=duration_seconds,
            engine_type=engine,
            negative_prompt=negative_prompt.strip() if negative_prompt else None,
            width=width,
            height=height,
            fps=fps,
            ltx_mode=mode,
            seed=seed,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    work_dir = record.work_dir
    if work_dir is None:
        raise HTTPException(status_code=503, detail="Job work directory was not created")

    if image is not None:
        content_type = (image.content_type or "").lower()
        suffix = ".png" if "png" in content_type else ".jpg" if "jpeg" in content_type else ".webp"
        input_path = work_dir / f"input{suffix}"
        input_path.write_bytes(data)
        target = work_dir / "input.png"
        if input_path != target:
            shutil.copy2(input_path, target)

    job_manager.enqueue(record.job_id)

    base = str(request.base_url).rstrip("/")
    return record.to_dict(base_url=base)


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
