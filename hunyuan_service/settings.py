"""Hunyuan engine settings (loads repo ``.env``)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from core.paths import HUNYUAN_SERVICE_ROOT, REPO_ROOT

DEFAULT_REPO = HUNYUAN_SERVICE_ROOT / "third_party" / "HunyuanVideo-1.5"
DEFAULT_CKPT = HUNYUAN_SERVICE_ROOT / "models" / "hunyuan" / "HunyuanVideo-1.5"


class HunyuanSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    HUNYUAN_PYTHON: str = ""
    HUNYUAN_ENV_DIR: str = "hunyuan_service/engines/hunyuan-env"
    HUNYUAN_REPO_DIR: str = str(DEFAULT_REPO)
    HUNYUAN_MODEL_PATH: str = str(DEFAULT_CKPT)
    HUNYUAN_RESOLUTION: str = "480p"
    HUNYUAN_ASPECT_RATIO: str = "16:9"
    HUNYUAN_DEVICE: str = "cuda"
    HUNYUAN_ENABLE_OFFLOAD: bool = True
    HUNYUAN_PROMPT_REWRITE: bool = False
    HUNYUAN_ENABLE_SR: bool = False
    HUNYUAN_ENABLE_STEP_DISTILL: bool = True
    HUNYUAN_STEP_DISTILL_STEPS: int = 8
    HUNYUAN_NUM_INFERENCE_STEPS: int = 50
    HUNYUAN_DEFAULT_FPS: int = 24
    HUNYUAN_DTYPE: str = "bf16"

    @property
    def project_root(self) -> Path:
        return REPO_ROOT

    def _resolve_path(self, raw: str) -> Path:
        path = Path(raw)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path

    @property
    def hunyuan_repo_dir(self) -> Path:
        return self._resolve_path(self.HUNYUAN_REPO_DIR)

    @property
    def hunyuan_model_path(self) -> Path:
        return self._resolve_path(self.HUNYUAN_MODEL_PATH)

    @property
    def hunyuan_env_dir(self) -> Path:
        return self._resolve_path(self.HUNYUAN_ENV_DIR)

    def resolved_hunyuan_python(self) -> str:
        if self.HUNYUAN_PYTHON.strip():
            return self.HUNYUAN_PYTHON.strip()
        dedicated = self.hunyuan_env_dir / "bin" / "python"
        if dedicated.is_file():
            return (
                str(dedicated.relative_to(REPO_ROOT))
                if dedicated.is_relative_to(REPO_ROOT)
                else str(dedicated)
            )
        for rel in (
            "cogvideox_service/engines/cogvideox-env/bin/python",
            "ltx_service/engines/ltx-env/bin/python",
            "venv/bin/python",
        ):
            p = REPO_ROOT / rel
            if p.is_file():
                return rel
        return os.environ.get("WAN_PYTHON", "python").strip() or "python"

    def resolved_cuda_device(self) -> str:
        return os.environ.get("WAN_CUDA_DEVICE", "0").strip() or "0"

    def resolved_resolution(self) -> str:
        r = (self.HUNYUAN_RESOLUTION or "480p").strip().lower()
        return r if r in ("480p", "720p") else "480p"

    def resolved_aspect_ratio(self) -> str:
        return (self.HUNYUAN_ASPECT_RATIO or "16:9").strip() or "16:9"


@lru_cache
def get_hunyuan_settings() -> HunyuanSettings:
    return HunyuanSettings()


def clear_hunyuan_settings_cache() -> None:
    get_hunyuan_settings.cache_clear()