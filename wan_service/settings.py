"""Wan TI2V paths and inference settings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.paths import REPO_ROOT, WAN_SERVICE_ROOT

DEFAULT_WAN_REPO = WAN_SERVICE_ROOT / "third_party" / "Wan2.2"
DEFAULT_CKPT = WAN_SERVICE_ROOT / "models" / "Wan2.2-TI2V-5B"

RuntimeMode = Literal["mac_dev", "linux_cuda"]


class WanSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    WAN_RUNTIME_MODE: str = ""
    WAN_INFERENCE_BACKEND: str = ""

    WAN_REPO_DIR: str = str(DEFAULT_WAN_REPO)
    WAN_CKPT_DIR: str = str(DEFAULT_CKPT)
    WAN_GENERATE_SCRIPT: str = ""
    WAN_PYTHON: str = "python"

    WAN_SIZE_LANDSCAPE: str = "1280*704"
    WAN_SIZE_PORTRAIT: str = "704*1280"
    WAN_INFERENCE_SAMPLE_STEPS: int | None = None
    WAN_INFERENCE_FRAME_NUM: int | None = None
    WAN_MIN_VRAM_GB: float = 14.0
    WAN_INFERENCE_GPU_ONLY: bool = True
    WAN_INFERENCE_FORCE_GPU_ONLY: bool = False
    WAN_GPU_ONLY_MIN_VRAM_GB: float = 24.0
    WAN_CUDA_DEVICE: str = "0"
    WAN_PERSISTENT_INFERENCE: bool = True
    WAN_WORKER_LOAD_TIMEOUT_SEC: float = 900.0

    @field_validator("WAN_RUNTIME_MODE", mode="before")
    @classmethod
    def normalize_runtime_mode(cls, value: str) -> str:
        return (value or "").strip().lower()

    @field_validator("WAN_RUNTIME_MODE")
    @classmethod
    def validate_runtime_mode(cls, value: str) -> str:
        if value and value not in ("mac_dev", "linux_cuda"):
            raise ValueError(
                f"Unsupported WAN_RUNTIME_MODE={value!r}. Use 'mac_dev' or 'linux_cuda'."
            )
        return value

    @field_validator("WAN_INFERENCE_BACKEND", mode="before")
    @classmethod
    def normalize_inference_backend(cls, value: str) -> str:
        return (value or "").strip().lower()

    @field_validator("WAN_INFERENCE_BACKEND")
    @classmethod
    def validate_inference_backend(cls, value: str) -> str:
        if value and value not in ("mac_mps", "linux_cuda"):
            raise ValueError(
                f"Unsupported WAN_INFERENCE_BACKEND={value!r}. Use 'mac_mps' or 'linux_cuda'."
            )
        return value

    def resolved_inference_backend(self) -> str:
        if self.WAN_INFERENCE_BACKEND in ("mac_mps", "linux_cuda"):
            return self.WAN_INFERENCE_BACKEND
        if self.resolved_runtime_mode() == "mac_dev":
            return "mac_mps"
        return "linux_cuda"

    @property
    def project_root(self) -> Path:
        return REPO_ROOT

    def _resolve_path(self, raw: str) -> Path:
        path = Path(raw)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path

    @property
    def wan_repo_dir(self) -> Path:
        return self._resolve_path(self.WAN_REPO_DIR)

    @property
    def wan_ckpt_dir(self) -> Path:
        return self._resolve_path(self.WAN_CKPT_DIR)

    @property
    def wan_generate_script(self) -> Path:
        if self.WAN_GENERATE_SCRIPT.strip():
            return self._resolve_path(self.WAN_GENERATE_SCRIPT)
        return self.wan_repo_dir / "generate.py"

    def resolved_runtime_mode(self) -> RuntimeMode:
        if self.WAN_RUNTIME_MODE in ("mac_dev", "linux_cuda"):
            return self.WAN_RUNTIME_MODE  # type: ignore[return-value]
        import platform

        if platform.system() == "Darwin":
            return "mac_dev"
        return "linux_cuda"
