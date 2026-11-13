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

    # Pre-job prompt rewrite (Ollama). Falls back to cinematic template if Ollama is down.
    LTX_PROMPT_ENHANCE: bool = True
    LTX_OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    LTX_OLLAMA_MODEL: str = "llama3.2:3b"
    LTX_OLLAMA_TIMEOUT_SEC: float = 60.0

    # I2V motion knobs (passed into upstream InferenceConfig).
    # Distilled 2B locks hard at strength≈1; lower strength + higher noise unlocks motion.
    LTX_CONDITIONING_STRENGTH: float = 0.7
    LTX_IMAGE_COND_NOISE_SCALE: float = 0.5
    LTX_DEFAULT_NEGATIVE_PROMPT: str = (
        "still image, frozen frame, no motion, static pose, slideshow, nearly still, "
        "subtle motion, micro movement, idle, posed photograph, "
        "worst quality, inconsistent motion, blurry, jittery, distorted, watermark"
    )
    # Default clip length when clients omit duration (more frames = smoother large moves).
    LTX_DEFAULT_DURATION_SECONDS: float = 5.0
    LTX_DEFAULT_FPS: int = 24
    # Post amplify deviation from first frame (1.0 = off). Distilled I2V needs ~2–2.5.
    LTX_MOTION_AMPLIFY: float = 2.4
    # Jobs longer than this are queued as sequential chunks with last-frame continuity.
    LTX_CHUNK_SECONDS: float = 10.0
    # Drop held last-frame tails (LTX often freezes at clip end / chunk seams).
    LTX_TRIM_TRAILING_FREEZE: bool = True
    LTX_TRIM_DIFF_THRESHOLD: float = 1.5
    LTX_TRIM_MIN_KEEP_RATIO: float = 0.75
    LTX_TRIM_MAX_FRAMES: int = 48
    # Extra frames to always shave from non-final chunks before bridge extract.
    LTX_CHUNK_MIN_TRIM_FRAMES: int = 12
    # Lighter always-trim on the final chunk so the ending does not hold a pose.
    LTX_FINAL_MIN_TRIM_FRAMES: int = 8
    # Continuation chunks (after 0): lock harder to previous last frame to keep quality.
    LTX_BRIDGE_CONDITIONING_STRENGTH: float = 0.88
    LTX_BRIDGE_IMAGE_COND_NOISE_SCALE: float = 0.2
    # Motion amplify on continuation chunks destroys fidelity — keep off when chunked.
    LTX_AMPLIFY_CONTINUATION_CHUNKS: bool = False

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
