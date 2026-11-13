"""LTX engine environment (isolated from wan_service settings)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from core.paths import LTX_SERVICE_ROOT, REPO_ROOT

DEFAULT_LTX_REPO = LTX_SERVICE_ROOT / "third_party" / "LTX-Video"

class LtxSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    LTX_PYTHON: str = ""
    LTX_ENV_DIR: str = "ltx_service/engines/ltx-env"
    LTX_REPO_DIR: str = str(DEFAULT_LTX_REPO)
    LTX_MODEL_PATH: str = "ltx_service/models/ltx/ltxv-2b-0.9.6-distilled-04-25.safetensors"
    LTX_CKPT_DIR: str = ""
    LTX_PIPELINE_CONFIG: str = "configs/ltxv-2b-0.9.6-distilled.yaml"

    @property
    def project_root(self) -> Path:
        return REPO_ROOT

    def _resolve_path(self, raw: str) -> Path:
        path = Path(raw)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path

    @property
    def ltx_repo_dir(self) -> Path:
        return self._resolve_path(self.LTX_REPO_DIR)

    @property
    def ltx_model_path(self) -> Path:
        return self._resolve_path(self.LTX_MODEL_PATH)

    @property
    def ltx_env_dir(self) -> Path:
        return self._resolve_path(self.LTX_ENV_DIR)

    @property
    def ltx_ckpt_dir(self) -> Path:
        if self.LTX_CKPT_DIR.strip():
            return self._resolve_path(self.LTX_CKPT_DIR)
        return self.ltx_model_path.parent

    def resolved_ltx_pipeline_config(self) -> str:
        raw = self.LTX_PIPELINE_CONFIG.strip() or "configs/ltxv-2b-0.9.6-distilled.yaml"
        path = Path(raw)
        if path.is_file():
            return str(path.resolve())
        candidate = self.ltx_repo_dir / raw
        return str(candidate) if candidate.is_file() else raw

    def resolved_ltx_python(self) -> str:
        if self.LTX_PYTHON.strip():
            return self.LTX_PYTHON.strip()
        return os.environ.get("WAN_PYTHON", "python").strip() or "python"

    def resolved_cuda_device(self) -> str:
        return os.environ.get("WAN_CUDA_DEVICE", "0").strip() or "0"


@lru_cache
def get_ltx_settings() -> LtxSettings:
    return LtxSettings()


def clear_ltx_settings_cache() -> None:
    get_ltx_settings.cache_clear()


ltx_settings = get_ltx_settings()
