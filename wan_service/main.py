"""
Wan 2.2 TI2V-5B image-to-video FastAPI service.

Run from repository root:
  uvicorn wan_service.main:app --host 127.0.0.1 --port 8080
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from pathlib import Path

import re

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from wan_service.utils.job_ids import require_job_id

from wan_service.core.config import settings
from wan_service.core.logging import configure_logging, get_logger
from wan_service.core.runtime import evaluate_inference_readiness
from wan_service.routers import api_router

logger = get_logger(__name__)

_JOB_PATH_RE = re.compile(
    r"^/(?:jobs|video/jobs|video/ws)/([^/]+)(?:/progress|/video)?/?$"
)


class RejectInvalidJobPollMiddleware(BaseHTTPMiddleware):
    """Return 400 before auth/routes for bad job ids (stops browser poll loops)."""

    async def dispatch(self, request: Request, call_next):
        if request.method != "GET":
            return await call_next(request)
        match = _JOB_PATH_RE.match(request.url.path)
        if not match:
            return await call_next(request)
        raw_id = match.group(1)
        try:
            require_job_id(raw_id)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        return await call_next(request)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    import asyncio

    from video_generation.progress_hub import progress_hub

    progress_hub.register_loop(asyncio.get_running_loop())
    configure_logging(settings.LOG_LEVEL)
    state = evaluate_inference_readiness()
    logger.info(
        "Started %s v%s | runtime_mode=%s | inference_ready=%s",
        settings.APP_NAME,
        settings.APP_VERSION,
        state.runtime_mode,
        state.ready,
    )
    if (
        settings.WAN_PERSISTENT_INFERENCE
        and settings.resolved_runtime_mode() == "linux_cuda"
        and state.ready
    ):
        from wan_service.apps.jobs.services.persistent_worker import (
            get_persistent_worker,
            shutdown_persistent_worker,
        )

        worker = get_persistent_worker()
        threading.Thread(
            target=worker.start,
            kwargs={"load_timeout_sec": settings.WAN_WORKER_LOAD_TIMEOUT_SEC},
            daemon=True,
            name="wan-persistent-worker-boot",
        ).start()
        logger.info(
            "Persistent inference worker loading in background (timeout=%ss)",
            settings.WAN_WORKER_LOAD_TIMEOUT_SEC,
        )
    try:
        yield
    finally:
        if settings.WAN_PERSISTENT_INFERENCE:
            from wan_service.apps.jobs.services.persistent_worker import (
                shutdown_persistent_worker,
            )

            shutdown_persistent_worker()


def create_app() -> FastAPI:
    application = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    application.add_middleware(RejectInvalidJobPollMiddleware)
    application.include_router(api_router)

    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.is_dir():
        job_loader = static_dir / "job_loader.html"

        @application.get("/ui/job_loader.html", include_in_schema=False)
        def _job_loader_html() -> Response:
            body = job_loader.read_bytes()
            return Response(
                content=body,
                media_type="text/html; charset=utf-8",
                headers={"Cache-Control": "no-store"},
            )

        application.mount("/ui", StaticFiles(directory=str(static_dir), html=True), name="ui")

        @application.get("/", include_in_schema=False)
        def _root_redirect() -> RedirectResponse:
            return RedirectResponse(url="/ui/job_loader.html")

    return application


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run(app, host=settings.WAN_WORKER_HOST, port=settings.WAN_WORKER_PORT)


if __name__ == "__main__":
    run()
