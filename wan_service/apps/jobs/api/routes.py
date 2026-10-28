"""Job submission and retrieval routes."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from wan_service import config as service_config
from wan_service.apps.jobs.services.manager import job_manager
from wan_service.core.runtime import evaluate_inference_readiness
from wan_service.utils.auth import require_worker_auth

router = APIRouter(prefix="/jobs", tags=["jobs"])

ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}


@router.post("")
async def create_job(
    request: Request,
    image: UploadFile = File(...),
    prompt: str = Form(...),
    duration_seconds: float = Form(5.0),
    _: None = Depends(require_worker_auth),
):
    if not prompt or not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")
    if len(prompt) > service_config.MAX_PROMPT_CHARS:
        raise HTTPException(status_code=400, detail="prompt too long")
    if duration_seconds <= 0 or duration_seconds > 60:
        raise HTTPException(status_code=400, detail="duration_seconds must be between 0 and 60")

    content_type = (image.content_type or "").lower()
    if content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported image type")

    readiness = evaluate_inference_readiness()
    if not readiness.ready:
        raise HTTPException(status_code=503, detail=readiness.public_reason)

    data = await image.read()
    if len(data) > service_config.MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image too large")
    if len(data) < 100:
        raise HTTPException(status_code=400, detail="Empty image")

    try:
        record = job_manager.create_job(prompt=prompt.strip(), duration_seconds=duration_seconds)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    work_dir = record.work_dir
    if work_dir is None:
        raise HTTPException(status_code=503, detail="Job work directory was not created")

    suffix = ".png" if "png" in content_type else ".jpg" if "jpeg" in content_type else ".webp"
    input_path = work_dir / f"input{suffix}"
    input_path.write_bytes(data)
    target = work_dir / "input.png"
    if input_path != target:
        shutil.copy2(input_path, target)

    job_manager.enqueue(record.job_id)

    base = str(request.base_url).rstrip("/")
    return record.to_dict(base_url=base)


@router.get("/{job_id}")
def get_job(job_id: str, request: Request, _: None = Depends(require_worker_auth)):
    record = job_manager.get(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    return record.to_dict(base_url=str(request.base_url).rstrip("/"))


@router.get("/{job_id}/video")
def get_job_video(job_id: str, _: None = Depends(require_worker_auth)):
    record = job_manager.get(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    if record.status != "completed" or not record.video_path:
        raise HTTPException(status_code=404, detail="Video not ready")
    path = Path(record.video_path).resolve()
    work_root = record.work_dir.resolve() if record.work_dir else None
    if work_root and work_root not in path.parents:
        raise HTTPException(status_code=400, detail="Invalid video path")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video file missing")
    return FileResponse(path, media_type="video/mp4", filename=f"{job_id}.mp4")
