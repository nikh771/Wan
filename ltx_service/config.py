"""LTX 2B distilled 0.9.6 — paths, frame/resolution helpers (no torch import)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

LTX_MODEL_ID = "ltxv-2b-0.9.6-distilled"
LTX_CHECKPOINT_FILENAME = "ltxv-2b-0.9.6-distilled-04-25.safetensors"
DEFAULT_PIPELINE_REL = "configs/ltxv-2b-0.9.6-distilled.yaml"
DEFAULT_NUM_INFERENCE_STEPS = 8
DEFAULT_GUIDANCE_SCALE = 1.0

LtxMode = str  # "text_to_video" | "image_to_video"


@dataclass(frozen=True)
class LtxFramePlan:
    requested_frames: int
    actual_frames: int
    actual_duration_seconds: float
    fps: int


@dataclass(frozen=True)
class LtxResolution:
    requested_width: int
    requested_height: int
    width: int
    height: int


def align_ltx_frame_count(raw_frames: int) -> int:
    """Official LTX frame rule: 8N + 1 (minimum 9). Round up to next valid count."""
    if raw_frames < 9:
        return 9
    n = math.ceil((raw_frames - 1) / 8)
    return int(n) * 8 + 1


def calculate_ltx_frame_count(duration_seconds: float, fps: int) -> LtxFramePlan:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    requested = max(1, int(round(duration_seconds * fps)))
    actual = align_ltx_frame_count(requested)
    return LtxFramePlan(
        requested_frames=requested,
        actual_frames=actual,
        actual_duration_seconds=actual / fps,
        fps=fps,
    )


def _round_up_32(value: int) -> int:
    if value <= 0:
        raise ValueError("dimension must be positive")
    return int(math.ceil(value / 32) * 32)


def normalize_ltx_resolution(
    width: int,
    height: int,
    *,
    max_width: int = 1280,
    max_height: int = 1280,
) -> LtxResolution:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    w, h = min(width, max_width), min(height, max_height)
    return LtxResolution(
        requested_width=width,
        requested_height=height,
        width=_round_up_32(w),
        height=_round_up_32(h),
    )


def default_model_path(project_root: Path, env_path: str) -> Path:
    if env_path.strip():
        p = Path(env_path.strip())
        return p if p.is_absolute() else (project_root / p)
    return project_root / "ltx_service" / "models" / "ltx" / LTX_CHECKPOINT_FILENAME


def materialize_pipeline_config(
    repo_dir: Path,
    pipeline_rel: str,
    checkpoint_path: Path,
    *,
    disable_prompt_enhancement: bool = False,
) -> Path:
    """Write resolved YAML with absolute checkpoint_path (does not modify repo file)."""
    import yaml

    src = pipeline_rel
    if not Path(src).is_absolute():
        src_path = repo_dir / pipeline_rel
    else:
        src_path = Path(pipeline_rel)
    if not src_path.is_file():
        raise FileNotFoundError(f"LTX pipeline config not found: {src_path}")

    data = yaml.safe_load(src_path.read_text(encoding="utf-8"))
    data["checkpoint_path"] = str(checkpoint_path.resolve())
    data["num_inference_steps"] = data.get("num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS)
    data["guidance_scale"] = data.get("guidance_scale", DEFAULT_GUIDANCE_SCALE)
    # Florence-2 + Llama enhancers will not fit alongside LTX+T5 on 24GB cards.
    if disable_prompt_enhancement:
        data["prompt_enhancement_words_threshold"] = 0
    out_dir = checkpoint_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / ".ltx-pipeline-resolved.yaml"
    dest.write_text(yaml.dump(data, sort_keys=False), encoding="utf-8")
    return dest
