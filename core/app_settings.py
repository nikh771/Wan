"""Shared API / worker settings (all engines)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from core.paths import REPO_ROOT


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "Wan 2.2 GPU Service"
    APP_VERSION: str = "0.1.0"
    LOG_LEVEL: str = "INFO"

    WAN_WORKER_API_KEY: str = ""
    WAN_WORKER_HOST: str = "0.0.0.0"
    WAN_WORKER_PORT: int = 8080

    WAN_JOBS_DIR: str = str(REPO_ROOT / "wan_jobs")
    WAN_MAX_QUEUE_SIZE: int = 8
    WAN_MAX_IMAGE_BYTES: int = 10 * 1024 * 1024
    WAN_MAX_PROMPT_CHARS: int = 8000

    FFMPEG_BINARY: str = "ffmpeg"
    FFPROBE_BINARY: str = "ffprobe"

    @property
    def project_root(self) -> Path:
        return REPO_ROOT

    @property
    def jobs_dir(self) -> Path:
        path = Path(self.WAN_JOBS_DIR)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path
