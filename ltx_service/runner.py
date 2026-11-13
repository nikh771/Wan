"""API-side LTX runner: spawns isolated LTX_PYTHON worker (never loads LTX in API process)."""

from __future__ import annotations

import json
import logging
import subprocess
import threading
from collections import deque
from collections.abc import Callable
from pathlib import Path

from ltx_service.config import (
    calculate_ltx_frame_count,
    normalize_ltx_resolution,
)
from ltx_service.settings import get_ltx_settings
from core.types import GenerationJobParams

logger = logging.getLogger(__name__)

_WORKER = Path(__file__).resolve().parent / "worker_main.py"


def _ltx_env() -> dict[str, str]:
    import os

    cfg = get_ltx_settings()
    env = os.environ.copy()
    roots = [str(cfg.project_root), str(cfg.ltx_repo_dir)]
    existing = env.get("PYTHONPATH")
    if existing:
        roots.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(roots)
    env["LTX_REPO_DIR"] = str(cfg.ltx_repo_dir)
    env["LTX_MODEL_PATH"] = str(cfg.ltx_model_path)
    env["LTX_PIPELINE_CONFIG"] = cfg.resolved_ltx_pipeline_config()
    env["CUDA_VISIBLE_DEVICES"] = cfg.resolved_cuda_device()
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _stream_output(
    process: subprocess.Popen[str],
    on_log_line: Callable[[str], None] | None,
) -> str:
    assert process.stdout is not None
    tail: deque[str] = deque(maxlen=50)

    def _reader() -> None:
        for line in process.stdout:
            s = line.rstrip()
            tail.append(s)
            logger.info("[ltx] %s", s)
            if on_log_line:
                on_log_line(s)

    threading.Thread(target=_reader, daemon=True).start()
    process.wait()
    return "\n".join(tail)


def run_ltx_generation(params: GenerationJobParams) -> dict:
    cfg = get_ltx_settings()
    if not _WORKER.is_file():
        raise FileNotFoundError(f"LTX worker missing: {_WORKER}")

    duration = float(params.duration_seconds or 5.0)
    width = params.width or 704
    height = params.height or 384
    mode = (params.mode or params.metadata.get("mode") or "image_to_video").strip()
    seed = int(params.seed if params.seed is not None else params.metadata.get("seed", -1))

    res = normalize_ltx_resolution(width, height)
    frames = calculate_ltx_frame_count(duration, params.fps)

    cmd = [
        cfg.resolved_ltx_python(),
        "-u",
        str(_WORKER),
        "--mode",
        mode,
        "--prompt",
        params.prompt,
        "--output",
        str(params.output_mp4.resolve()),
        "--width",
        str(res.width),
        "--height",
        str(res.height),
        "--fps",
        str(params.fps),
        "--duration-seconds",
        str(duration),
        "--seed",
        str(seed),
    ]
    if params.negative_prompt:
        cmd.extend(["--negative-prompt", params.negative_prompt])
    if mode == "image_to_video":
        if params.image_path is None:
            raise ValueError("image_to_video requires an input image")
        cmd.extend(["--image", str(params.image_path.resolve())])

    if params.on_log_line:
        params.on_log_line(
            f"[ltx_service] LTX {mode} | {res.width}x{res.height} "
            f"frames={frames.actual_frames} (~{frames.actual_duration_seconds:.2f}s)"
        )

    proc = subprocess.Popen(
        cmd,
        cwd=str(cfg.project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_ltx_env(),
        start_new_session=True,
    )
    tail = _stream_output(proc, params.on_log_line)
    if proc.returncode != 0:
        raise RuntimeError(f"LTX worker exit {proc.returncode}\n{tail[-2000:]}")

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
                raise RuntimeError(payload.get("message") or "LTX error")
    return {"engine": "ltx", "mode": mode}
