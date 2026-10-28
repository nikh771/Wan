"""
Wan 2.2 TI2V-5B image-to-video FastAPI service.

Run from repository root:
  uvicorn wan_service.main:app --host 127.0.0.1 --port 8080
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

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
    yield


def create_app() -> FastAPI:
    application = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    application.include_router(api_router)
    return application


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run(app, host=settings.WAN_WORKER_HOST, port=settings.WAN_WORKER_PORT)


if __name__ == "__main__":
    run()
