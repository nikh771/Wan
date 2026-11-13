"""CogVideoX engine environment (isolated from wan_service / ltx_service settings)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from core.paths import COGVIDEOX_SERVICE_ROOT, REPO_ROOT


class CogVideoXSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    COGVIDEOX_PYTHON: str = ""
    COGVIDEOX_ENV_DIR: str = "cogvideox_service/engines/cogvideox-env"
    COGVIDEOX_MODEL_PATH: str = "cogvideox_service/models/cogvideox/CogVideoX-2b"
    COGVIDEOX_HF_ID: str = "zai-org/CogVideoX-2b"
    COGVIDEOX_DTYPE: str = "float16"
    # memory | normal — memory uses model CPU offload + VAE tiling (recommended on 24GB).
    COGVIDEOX_MEMORY_MODE: str = "memory"
    COGVIDEOX_DEFAULT_WIDTH: int = 720
    COGVIDEOX_DEFAULT_HEIGHT: int = 480
    COGVIDEOX_DEFAULT_NUM_FRAMES: int = 49
    COGVIDEOX_DEFAULT_FPS: int = 8
    COGVIDEOX_DEFAULT_STEPS: int = 50
    COGVIDEOX_DEFAULT_GUIDANCE_SCALE: float = 6.0
    COGVIDEOX_DEFAULT_SEED: int = 12345
    COGVIDEOX_DEFAULT_DURATION_SECONDS: float = 6.125
    COGVIDEOX_CHUNK_SECONDS: float = 10.0

    @property
    def project_root(self) -> Path:
        return REPO_ROOT

    def _resolve_path(self, raw: str) -> Path:
        path = Path(raw)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path

    @property
    def cogvideox_model_path(self) -> Path:
        return self._resolve_path(self.COGVIDEOX_MODEL_PATH)

    @property
    def cogvideox_env_dir(self) -> Path:
        return self._resolve_path(self.COGVIDEOX_ENV_DIR)

    def resolved_cogvideox_python(self) -> str:
        if self.COGVIDEOX_PYTHON.strip():
            return self.COGVIDEOX_PYTHON.strip()
        # Prefer dedicated cogvideox-env (same isolation pattern as LTX).
        dedicated = self.cogvideox_env_dir / "bin" / "python"
        if dedicated.is_file():
            return (
                str(dedicated.relative_to(REPO_ROOT))
                if dedicated.is_relative_to(REPO_ROOT)
                else str(dedicated)
            )
        # Fallbacks only for worker Python — package/models stay under cogvideox_service/.
        ltx_fallback = REPO_ROOT / "ltx_service" / "engines" / "ltx-env" / "bin" / "python"
        if ltx_fallback.is_file():
            return "ltx_service/engines/ltx-env/bin/python"
        wan_venv = REPO_ROOT / "venv" / "bin" / "python"
        if wan_venv.is_file():
            return "venv/bin/python"
        return os.environ.get("WAN_PYTHON", "python").strip() or "python"

    def resolved_cuda_device(self) -> str:
        return os.environ.get("WAN_CUDA_DEVICE", "0").strip() or "0"

    def resolved_memory_mode(self) -> str:
        mode = (self.COGVIDEOX_MEMORY_MODE or "memory").strip().lower()
        if mode not in ("memory", "normal"):
            return "memory"
        return mode


@lru_cache
def get_cogvideox_settings() -> CogVideoXSettings:
    return CogVideoXSettings()


def clear_cogvideox_settings_cache() -> None:
    get_cogvideox_settings.cache_clear()


cogvideox_settings = get_cogvideox_settings()
