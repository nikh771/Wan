"""Application settings (environment variables)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WAN_REPO = PROJECT_ROOT / "third_party" / "Wan2.2"
DEFAULT_CKPT = PROJECT_ROOT / "models" / "Wan2.2-TI2V-5B"

RuntimeMode = Literal["mac_dev", "linux_cuda"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "Wan 2.2 GPU Service"
    APP_VERSION: str = "0.1.0"
    LOG_LEVEL: str = "INFO"

    # mac_dev: API + tests; linux_cuda: NVIDIA inference host
    WAN_RUNTIME_MODE: str = ""
    # mac_mps | linux_cuda — passed to upstream generate.py (WAN_INFERENCE_BACKEND)
    WAN_INFERENCE_BACKEND: str = ""

    WAN_REPO_DIR: str = str(DEFAULT_WAN_REPO)
    WAN_CKPT_DIR: str = str(DEFAULT_CKPT)
    WAN_GENERATE_SCRIPT: str = ""
    WAN_PYTHON: str = "python"

    WAN_WORKER_API_KEY: str = ""
    WAN_WORKER_HOST: str = "0.0.0.0"
    WAN_WORKER_PORT: int = 8080

    WAN_JOBS_DIR: str = str(PROJECT_ROOT / "wan_jobs")
    WAN_MAX_QUEUE_SIZE: int = 8
    WAN_MAX_IMAGE_BYTES: int = 10 * 1024 * 1024
    WAN_MAX_PROMPT_CHARS: int = 8000

    FFMPEG_BINARY: str = "ffmpeg"
    FFPROBE_BINARY: str = "ffprobe"

    WAN_SIZE_LANDSCAPE: str = "1280*704"
    WAN_SIZE_PORTRAIT: str = "704*1280"
    # Override generate.py defaults (ti2v-5B: 50 steps, 121 frames). Lower = faster, lower quality.
    WAN_INFERENCE_SAMPLE_STEPS: int | None = None
    WAN_INFERENCE_FRAME_NUM: int | None = None

    # Heuristic only — does not guarantee TI2V-5B will succeed (linux_cuda).
    WAN_MIN_VRAM_GB: float = 14.0

    # Prefer full-GPU inference when VRAM allows (see WAN_GPU_ONLY_MIN_VRAM_GB).
    WAN_INFERENCE_GPU_ONLY: bool = True
    # Skip VRAM threshold; always pass full-GPU flags to generate.py (may OOM on small GPUs).
    WAN_INFERENCE_FORCE_GPU_ONLY: bool = False
    WAN_GPU_ONLY_MIN_VRAM_GB: float = 24.0
    WAN_CUDA_DEVICE: str = "0"

    # Keep TI2V weights loaded in a background worker (faster from job 2 onward).
    WAN_PERSISTENT_INFERENCE: bool = True
    # Max seconds to wait for the worker to finish loading checkpoints on startup.
    WAN_WORKER_LOAD_TIMEOUT_SEC: float = 900.0

    # LTX-Video 2B (separate venv — see requirements-ltx-gpu.txt)
    LTX_PYTHON: str = ""
    LTX_REPO_DIR: str = str(PROJECT_ROOT / "third_party" / "LTX-Video")
    LTX_CKPT_DIR: str = ""
    LTX_PIPELINE_CONFIG: str = "configs/ltxv-2b-0.9.6-dev.yaml"

    # Storyboard defaults when width/height omitted on create-job
    STORYBOARD_DEFAULT_FPS: int = 24

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
        return PROJECT_ROOT

    def _resolve_path(self, raw: str) -> Path:
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
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

    @property
    def ltx_repo_dir(self) -> Path:
        return self._resolve_path(self.LTX_REPO_DIR)

    @property
    def ltx_ckpt_dir(self) -> Path:
        if self.LTX_CKPT_DIR.strip():
            return self._resolve_path(self.LTX_CKPT_DIR)
        return self.ltx_repo_dir

    def resolved_ltx_python(self) -> str:
        if self.LTX_PYTHON.strip():
            return self.LTX_PYTHON.strip()
        return self.WAN_PYTHON

    @property
    def jobs_dir(self) -> Path:
        path = Path(self.WAN_JOBS_DIR)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    def resolved_runtime_mode(self) -> RuntimeMode:
        if self.WAN_RUNTIME_MODE in ("mac_dev", "linux_cuda"):
            return self.WAN_RUNTIME_MODE  # type: ignore[return-value]
        import platform

        if platform.system() == "Darwin":
            return "mac_dev"
        return "linux_cuda"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def clear_settings_cache() -> None:
    get_settings.cache_clear()


settings = get_settings()
