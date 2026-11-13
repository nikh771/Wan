"""Single video generation API: POST /video/generate with mode t2v | i2v."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from cogvideox_service.health import cogvideox_health_payload
from ltx_service.health import ltx_health_payload
from video_generation.progress_hub import progress_hub, stream_job_events
from video_generation.service import persist_image_and_enqueue, submit_video_job
from wan_service import config as service_config
from wan_service.apps.jobs.services.manager import job_manager
from wan_service.core.runtime import evaluate_inference_readiness
from wan_service.utils.auth import require_worker_auth
from wan_service.utils.job_ids import require_job_id

router = APIRouter(prefix="/video", tags=["video_generation"])


def _job_not_found(job_id: str) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail={"message": "Job not found", "job_id": job_id, "revoke_polling": True},
    )


def _websocket_url(request: Request, job_id: str) -> str:
    scheme = "wss" if request.url.scheme == "https" else "ws"
    return f"{scheme}://{request.url.netloc}/video/ws/{job_id}"


def _public_job_created(
    *,
    request: Request,
    record,
    mode: str,
    estimated_seconds: float,
    queue_ahead: int,
) -> dict:
    base = str(request.base_url).rstrip("/")
    return {
        "job_id": record.job_id,
        "status": record.status,
        "mode": mode,
        "engine_type": record.engine_type,
        "estimated_total_seconds": estimated_seconds,
        "estimated_remaining_seconds": estimated_seconds,
        "queue_ahead": queue_ahead,
        "progress_url": f"{base}/video/jobs/{record.job_id}",
        "websocket_url": _websocket_url(request, record.job_id),
        "video_url": None,
    }


@router.post("/generate")
async def generate_video(
    request: Request,
    mode: str = Form(..., description="t2v | i2v (or text_to_video | image_to_video)"),
    prompt: str = Form(...),
    duration_seconds: float = Form(5.0),
    engine_type: str = Form("wan"),
    image: Annotated[UploadFile | None, File()] = None,
    negative_prompt: str | None = Form(None),
    width: int | None = Form(None),
    height: int | None = Form(None),
    fps: int | None = Form(None),
    seed: int | None = Form(None),
    _: None = Depends(require_worker_auth),
):
    ctx = submit_video_job(
        mode=mode,
        prompt=prompt,
        duration_seconds=duration_seconds,
        engine_type=engine_type,
        image=image,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        fps=fps,
        seed=seed,
    )
    record = ctx["record"]
    if image is not None:
        await persist_image_and_enqueue(
            record.job_id,
            ctx["image"],
            ctx["content_type"],
        )
    else:
        job_manager.enqueue(record.job_id)
    return _public_job_created(
        request=request,
        record=record,
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


@router.get("/engines")
def list_engines(request: Request) -> dict:
    require_worker_auth(request)
    wan = evaluate_inference_readiness()
    return {
        "engines": {
            "wan": {
                "ready": wan.ready,
                "detail": wan.public_reason,
            },
            "ltx": ltx_health_payload(),
            "cogvideox": cogvideox_health_payload(),
        }
    }

