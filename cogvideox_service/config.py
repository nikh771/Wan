"""CogVideoX-2B — paths, frame/resolution helpers (no torch import)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

COGVIDEOX_MODEL_ID = "CogVideoX-2b"
COGVIDEOX_HF_ID = "zai-org/CogVideoX-2b"
COGVIDEOX_LICENSE = "Apache-2.0"

# Official Diffusers / model-card defaults for CogVideoX-2B T2V.
DEFAULT_WIDTH = 720
DEFAULT_HEIGHT = 480
DEFAULT_NUM_FRAMES = 49
DEFAULT_FPS = 8
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 6.0
DEFAULT_SEED = 12345

# CogVideoX temporal rule: frames = 8N + 1 (e.g. 49).
MIN_FRAMES = 9


@dataclass(frozen=True)
class CogVideoXFramePlan:
    requested_frames: int
    actual_frames: int
    actual_duration_seconds: float
    fps: int


@dataclass(frozen=True)
class CogVideoXResolution:
    requested_width: int
    requested_height: int
    width: int
    height: int


def align_cogvideox_frame_count(raw_frames: int) -> int:
    """Official CogVideoX frame rule: 8N + 1 (minimum 9). Round up to next valid count."""
    if raw_frames < MIN_FRAMES:
        return MIN_FRAMES
    n = math.ceil((raw_frames - 1) / 8)
    return int(n) * 8 + 1


def calculate_cogvideox_frame_count(
    *,
    num_frames: int | None = None,
    duration_seconds: float | None = None,
    fps: int = DEFAULT_FPS,
) -> CogVideoXFramePlan:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if num_frames is not None:
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        requested = int(num_frames)
    elif duration_seconds is not None:
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        requested = max(1, int(round(duration_seconds * fps)))
    else:
        requested = DEFAULT_NUM_FRAMES
    actual = align_cogvideox_frame_count(requested)
    return CogVideoXFramePlan(
        requested_frames=requested,
        actual_frames=actual,
        actual_duration_seconds=actual / fps,
        fps=fps,
    )


def calculate_cogvideox_duration(num_frames: int, fps: int) -> float:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    return float(num_frames) / float(fps)


def _round_up_16(value: int) -> int:
    if value <= 0:
        raise ValueError("dimension must be positive")
    return int(math.ceil(value / 16) * 16)


def normalize_cogvideox_resolution(
    width: int,
    height: int,
    *,
    max_width: int = 1360,
    max_height: int = 768,
) -> CogVideoXResolution:
    """Width/height must be divisible by 16 for CogVideoX."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    w, h = min(width, max_width), min(height, max_height)
    return CogVideoXResolution(
        requested_width=width,
        requested_height=height,
        width=_round_up_16(w),
        height=_round_up_16(h),
    )


def default_model_path(project_root: Path, env_path: str = "") -> Path:
    if env_path.strip():
        p = Path(env_path.strip())
        return p if p.is_absolute() else (project_root / p)
    return (
        project_root
        / "cogvideox_service"
        / "models"
        / "cogvideox"
        / "CogVideoX-2b"
    )


def model_available(model_dir: Path) -> bool:
    """True when a local Diffusers snapshot looks complete enough to load."""
    if not model_dir.is_dir():
        return False
    index = model_dir / "model_index.json"
    transformer = model_dir / "transformer"
    text_encoder = model_dir / "text_encoder"
    vae = model_dir / "vae"
    return (
        index.is_file()
        and transformer.is_dir()
        and text_encoder.is_dir()
        and vae.is_dir()
    )


LICENSE_METADATA = {
    "model": "CogVideoX-2B",
    "code_license": "Apache-2.0",
    "model_license": "Apache-2.0",
    "commercial_use": True,
    "hf_id": COGVIDEOX_HF_ID,
    "note": (
        "CogVideoX-2B model, Transformers module and VAE are Apache-2.0. "
        "CogVideoX-5B has different licensing — do not conflate."
    ),
}
