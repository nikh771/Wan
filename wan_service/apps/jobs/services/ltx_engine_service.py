"""LTX-Video 2B engine — isolated LTX_PYTHON subprocess."""

from __future__ import annotations

import json
import logging
import subprocess
import threading
from collections import deque
from collections.abc import Callable
from pathlib import Path

from wan_service.core.config import get_settings
from wan_service.engines.types import GenerationJobParams

logger = logging.getLogger(__name__)

_WORKER_SCRIPT = (
    Path(__file__).resolve().parents[1] / "ltx_worker_process.py"
)


def _ltx_env() -> dict[str, str]:
    import os

    cfg = get_settings()
    env = os.environ.copy()
    roots = [str(cfg.project_root)]
    if cfg.LTX_REPO_DIR.strip():
        roots.append(str(cfg.ltx_repo_dir))
    existing = env.get("PYTHONPATH")
    if existing:
        roots.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(roots)
    env["LTX_REPO_DIR"] = str(cfg.ltx_repo_dir)
    if cfg.LTX_CKPT_DIR.strip():
        env["LTX_CKPT_DIR"] = str(cfg.ltx_ckpt_dir)
    if cfg.LTX_PIPELINE_CONFIG.strip():
        env["LTX_PIPELINE_CONFIG"] = cfg.LTX_PIPELINE_CONFIG
    env["CUDA_VISIBLE_DEVICES"] = cfg.WAN_CUDA_DEVICE.strip() or "0"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _stream_output(
    process: subprocess.Popen[str],
    *,
    on_log_line: Callable[[str], None] | None,
) -> str:
    assert process.stdout is not None
    tail: deque[str] = deque(maxlen=30)

    def _reader() -> None:
        for line in process.stdout:
            stripped = line.rstrip()
            tail.append(stripped)
            logger.info("[ltx worker] %s", stripped)
            if on_log_line:
                on_log_line(stripped)

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    process.wait()
    thread.join(timeout=1)
    return "\n".join(tail)


def run_ltx_engine(params: GenerationJobParams) -> dict:
    cfg = get_settings()
    python = cfg.resolved_ltx_python()
    if not _WORKER_SCRIPT.is_file():
        raise FileNotFoundError(f"LTX worker script missing: {_WORKER_SCRIPT}")

    duration = float(params.duration_seconds or 2.0)
    width = params.width or 704
    height = params.height or 1280

    cmd = [
        python,
        "-u",
        str(_WORKER_SCRIPT),
        "--image",
        str(params.image_path.resolve()),
        "--prompt",
        params.prompt,
        "--output",
        str(params.output_mp4.resolve()),
        "--width",
        str(width),
        "--height",
        str(height),
        "--fps",
        str(params.fps),
        "--duration-seconds",
        str(duration),
    ]
    if params.negative_prompt:
        cmd.extend(["--negative-prompt", params.negative_prompt])

    if params.on_log_line:
        params.on_log_line(f"[wan_service] Starting LTX worker (size={width}x{height})")

    process = subprocess.Popen(
        cmd,
        cwd=str(cfg.project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_ltx_env(),
        start_new_session=True,
    )
    log_tail = _stream_output(process, on_log_line=params.on_log_line)
    if process.returncode != 0:
        raise RuntimeError(
            f"LTX worker failed (exit {process.returncode}). Output:\n{log_tail[-1500:]}"
        )

    for line in reversed(log_tail.splitlines()):
        line = line.strip()
        if line.startswith("{") and '"status"' in line:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("status") == "ok":
                return payload.get("metadata") or {}
            if payload.get("status") == "error":
                raise RuntimeError(payload.get("message") or "LTX worker error")
    return {"engine": "ltx"}
