"""Single video generation API: POST /video/generate with mode t2v | i2v."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import ValidationError

from video_generation.progress_hub import progress_hub, stream_job_events
from video_generation.schemas import VideoGenerateRequest
from video_generation.service import (
    finalize_job_enqueue,
    public_job_created_payload,
    submit_video_job,
    submit_video_job_from_request,
)
from wan_service import config as service_config
from wan_service.apps.jobs.services.manager import job_manager
from wan_service.utils.auth import require_worker_auth
from wan_service.utils.job_ids import require_job_id
router = APIRouter(prefix="/video", tags=["video_generation"])


def _job_not_found(job_id: str) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail={"message": "Job not found", "job_id": job_id, "revoke_polling": True},
    )


@router.post("/generate")
async def generate_video(
    request: Request,
    _: None = Depends(require_worker_auth),
    # Multipart (legacy / UI)
    mode: str | None = Form(None),
    prompt: str | None = Form(None),
    duration_seconds: float | None = Form(None),
    engine_type: str | None = Form(None),
    image: Annotated[UploadFile | None, File()] = None,
    negative_prompt: str | None = Form(None),
    width: int | None = Form(None),
    height: int | None = Form(None),
    fps: int | None = Form(None),
    seed: int | None = Form(None),
):
    content_type = (request.headers.get("content-type") or "").lower()
    body_model: VideoGenerateRequest | None = None

    if "application/json" in content_type:
        try:
            raw = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
        try:
            body_model = VideoGenerateRequest.model_validate(raw)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        ctx = submit_video_job_from_request(body_model, image=image)
    elif mode is not None and prompt is not None:
        ctx = submit_video_job(
            mode=mode,
            prompt=prompt,
            duration_seconds=float(duration_seconds if duration_seconds is not None else 5.0),
            engine_type=engine_type or "wan",
            image=image,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            fps=fps,
            seed=seed,
        )
    else:
        raise HTTPException(
            status_code=400,
            detail="Send application/json (video plan) or multipart form with mode and prompt",
        )

    await finalize_job_enqueue(ctx, image=image, body=body_model)
    return public_job_created_payload(
        request=request,
        record=ctx["record"],
        mode=ctx["mode"],
        estimated_seconds=ctx["estimated_seconds"],
        queue_ahead=ctx["queue_ahead"],
    )


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request):
    job_id = require_job_id(job_id)
    require_worker_auth(request)
    record = job_manager.get(job_id)
    if not record:
        raise _job_not_found(job_id)
    return record.to_progress_dict(base_url=str(request.base_url).rstrip("/"))


@router.get("/jobs/{job_id}/video")
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


@router.websocket("/ws/{job_id}")
async def job_progress_websocket(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    try:
        job_id = require_job_id(job_id)
    except HTTPException:
        await websocket.close(code=4400)
        return

    api_key = websocket.headers.get("authorization", "")
    if api_key.lower().startswith("bearer "):
        api_key = api_key[7:].strip()
    elif websocket.query_params.get("api_key"):
        api_key = websocket.query_params.get("api_key", "").strip()
    if not service_config.WORKER_API_KEY or api_key != service_config.WORKER_API_KEY:
        await websocket.close(code=4401)
        return

    if job_manager.get(job_id) is None:
        await websocket.close(code=4404)
        return

    def poll_fallback() -> dict | None:
        record = job_manager.get(job_id)
        if record is None:
            return {"type": "progress", "job_id": job_id, "done": True, "status": "failed"}
        return record.to_ws_event()

    try:
        await stream_job_events(websocket, job_id, poll_fallback=poll_fallback)
    except WebSocketDisconnect:
        return

