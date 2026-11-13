"""API-side CogVideoX runner: spawns isolated COGVIDEOX_PYTHON worker."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from collections import deque
from collections.abc import Callable
from pathlib import Path

from core.types import GenerationJobParams
from cogvideox_service.config import (
    DEFAULT_FPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_NUM_FRAMES,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_WIDTH,
    calculate_cogvideox_frame_count,
    normalize_cogvideox_resolution,
)
from cogvideox_service.settings import get_cogvideox_settings

logger = logging.getLogger(__name__)

_WORKER = Path(__file__).resolve().parent / "worker_main.py"


def _cogvideox_env() -> dict[str, str]:
    cfg = get_cogvideox_settings()
    env = os.environ.copy()
    roots = [str(cfg.project_root)]
    existing = env.get("PYTHONPATH")
    if existing:
        roots.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(roots)
    env["COGVIDEOX_MODEL_PATH"] = str(cfg.cogvideox_model_path)
    env["COGVIDEOX_HF_ID"] = cfg.COGVIDEOX_HF_ID
    env["COGVIDEOX_DTYPE"] = cfg.COGVIDEOX_DTYPE
    env["COGVIDEOX_MEMORY_MODE"] = cfg.resolved_memory_mode()
    env["CUDA_VISIBLE_DEVICES"] = cfg.resolved_cuda_device()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["HF_HUB_ENABLE_HF_TRANSFER"] = env.get("HF_HUB_ENABLE_HF_TRANSFER", "0")
    return env


def _stream_output(
    process: subprocess.Popen[str],
    on_log_line: Callable[[str], None] | None,
) -> str:
    assert process.stdout is not None
    tail: deque[str] = deque(maxlen=80)

    def _reader() -> None:
        for line in process.stdout:
            s = line.rstrip()
            tail.append(s)
            logger.info("[cogvideox] %s", s)
            if on_log_line:
                on_log_line(s)

    threading.Thread(target=_reader, daemon=True).start()
    process.wait()
    return "\n".join(tail)


def _parse_worker_ok(tail: str) -> dict:
    for line in reversed(tail.splitlines()):
        line = line.strip()
        if line.startswith("{") and '"status"' in line:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("status") == "ok":
                return payload.get("result") or {}
            if payload.get("status") == "error":
                raise RuntimeError(payload.get("message") or "CogVideoX error")
    return {}


def run_cogvideox_generation(params: GenerationJobParams) -> dict:
    cfg = get_cogvideox_settings()
    if not _WORKER.is_file():
        raise FileNotFoundError(f"CogVideoX worker missing: {_WORKER}")

    mode = (params.mode or params.metadata.get("mode") or "text_to_video").strip()
    if mode != "text_to_video":
        raise ValueError(
            "CogVideoX POC supports text_to_video only. "
            "I2V needs a separate I2V checkpoint."
        )

    width = params.width or cfg.COGVIDEOX_DEFAULT_WIDTH or DEFAULT_WIDTH
    height = params.height or cfg.COGVIDEOX_DEFAULT_HEIGHT or DEFAULT_HEIGHT
    fps = int(params.fps or cfg.COGVIDEOX_DEFAULT_FPS or DEFAULT_FPS)
    seed = int(
        params.seed
        if params.seed is not None
        else params.metadata.get("seed", cfg.COGVIDEOX_DEFAULT_SEED)
    )
    steps = int(params.metadata.get("steps", cfg.COGVIDEOX_DEFAULT_STEPS or DEFAULT_NUM_INFERENCE_STEPS))
    guidance = float(
        params.metadata.get(
            "guidance_scale",
            cfg.COGVIDEOX_DEFAULT_GUIDANCE_SCALE or DEFAULT_GUIDANCE_SCALE,
        )
    )
    memory_mode = str(
        params.metadata.get("memory_mode", cfg.resolved_memory_mode())
    ).strip().lower()
    if memory_mode not in ("memory", "normal"):
        memory_mode = "memory"

    # Prefer explicit num_frames from metadata; else derive from duration; else default 49.
    num_frames_meta = params.metadata.get("num_frames")
    if num_frames_meta is not None:
        frame_plan = calculate_cogvideox_frame_count(
            num_frames=int(num_frames_meta), fps=fps
        )
    elif params.duration_seconds is not None and params.duration_seconds > 0:
        frame_plan = calculate_cogvideox_frame_count(
            duration_seconds=float(params.duration_seconds), fps=fps
        )
    else:
        frame_plan = calculate_cogvideox_frame_count(
            num_frames=int(cfg.COGVIDEOX_DEFAULT_NUM_FRAMES or DEFAULT_NUM_FRAMES),
            fps=fps,
        )

    res = normalize_cogvideox_resolution(width, height)
    if params.on_log_line:
        params.on_log_line(
            f"[cogvideox_service] T2V | {res.width}x{res.height} "
            f"frames={frame_plan.actual_frames} "
            f"(~{frame_plan.actual_duration_seconds:.3f}s @ {fps}fps) "
            f"steps={steps} guidance={guidance} seed={seed} memory_mode={memory_mode}"
        )

    params.output_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        cfg.resolved_cogvideox_python(),
        "-u",
        str(_WORKER),
        "--mode",
        mode,
        "--prompt",
        params.prompt,
        "--negative-prompt",
        params.negative_prompt or "",
        "--output",
        str(params.output_mp4.resolve()),
        "--width",
        str(res.width),
        "--height",
        str(res.height),
        "--fps",
        str(fps),
        "--num-frames",
        str(frame_plan.actual_frames),
        "--steps",
        str(steps),
        "--guidance-scale",
        str(guidance),
        "--seed",
        str(seed),
        "--memory-mode",
        memory_mode,
    ]

    py = cfg.resolved_cogvideox_python()
    py_path = Path(py) if Path(py).is_absolute() else cfg.project_root / py
    if not py_path.is_file():
        raise FileNotFoundError(f"COGVIDEOX_PYTHON not found: {py}")

    proc = subprocess.Popen(
        cmd,
        cwd=str(cfg.project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_cogvideox_env(),
        start_new_session=True,
    )
    tail = _stream_output(proc, params.on_log_line)
    if proc.returncode != 0:
        raise RuntimeError(f"CogVideoX worker exit {proc.returncode}\n{tail[-2000:]}")
    result = _parse_worker_ok(tail)
    result.setdefault("engine", "cogvideox")
    result.setdefault("status", "completed")
    result.setdefault("output_path", str(params.output_mp4))
    return result
