"""
Wan 2.2 TI2V-5B image-to-video FastAPI service.

Run from repository root:
  uvicorn wan_service.main:app --host 127.0.0.1 --port 8080
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from wan_service.core.config import settings
from wan_service.core.logging import configure_logging, get_logger
from wan_service.core.runtime import evaluate_inference_readiness
from wan_service.routers import api_router

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
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
    application.include_router(api_router)

    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.is_dir():
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
