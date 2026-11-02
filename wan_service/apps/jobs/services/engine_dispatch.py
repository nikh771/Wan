"""Route a generation job to Wan, LTX, or Storyboard."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from wan_service.apps.jobs.services.ltx_engine_service import run_ltx_engine
from wan_service.apps.jobs.services.storyboard_engine_service import run_storyboard_engine
from wan_service.apps.jobs.services.wan_engine_service import run_wan_engine
from wan_service.core.config import get_settings
from wan_service.core.runtime import evaluate_inference_readiness, probe_ffmpeg
from wan_service.engines.types import EngineType, GenerationJobParams


@dataclass(frozen=True)
class EngineReadiness:
    ready: bool
    public_reason: str


def normalize_engine(value: str) -> EngineType:
    engine = (value or "wan").strip().lower()
    if engine in ("wan", "ltx", "storyboard"):
        return engine  # type: ignore[return-value]
    raise ValueError(
        f"Unsupported engine_type {value!r}. Use wan, ltx, or storyboard."
    )


def _ltx_python_executable() -> Path | None:
    cfg = get_settings()
    raw = cfg.resolved_ltx_python()
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = cfg.project_root / candidate
    if candidate.is_file():
        return candidate
    found = shutil.which(raw)
    return Path(found) if found else None


def evaluate_engine_readiness(engine: EngineType) -> EngineReadiness:
    if engine == "wan":
        state = evaluate_inference_readiness()
        return EngineReadiness(state.ready, state.public_reason)

    if engine == "storyboard":
        if not probe_ffmpeg():
            return EngineReadiness(
                False,
                "FFmpeg/ffprobe required for storyboard rendering.",
            )
        try:
            import cv2  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError:
            return EngineReadiness(
                False,
                "Storyboard requires opencv-python and Pillow in the API venv.",
            )
        return EngineReadiness(True, "Storyboard engine ready.")

    ltx_py = _ltx_python_executable()
    if ltx_py is None:
        cfg = get_settings()
        return EngineReadiness(
            False,
            f"LTX_PYTHON not found: {cfg.resolved_ltx_python()!r}",
        )
    repo = get_settings().ltx_repo_dir
    if not (repo / "inference.py").is_file():
        return EngineReadiness(
            False,
            f"LTX repo missing at {repo}. Run scripts/runpod_remote_setup_ltx.sh",
        )
    if not probe_ffmpeg():
        return EngineReadiness(False, "FFmpeg required for LTX output handling.")
    return EngineReadiness(True, "LTX engine ready.")


def assert_engine_available(engine: EngineType) -> None:
    readiness = evaluate_engine_readiness(engine)
    if not readiness.ready:
        raise RuntimeError(readiness.public_reason)


def run_engine(params: GenerationJobParams, engine: EngineType) -> dict:
    if engine == "wan":
        run_wan_engine(params)
        return {"engine": "wan", "engine_used": "wan"}
    if engine == "storyboard":
        meta = run_storyboard_engine(params)
        meta["engine_used"] = "storyboard"
        return meta
    meta = run_ltx_engine(params)
    meta["engine_used"] = "ltx"
    return meta
