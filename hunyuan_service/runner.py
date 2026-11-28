"""API-side Hunyuan runner: subprocess to upstream ``generate.py``."""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from collections import deque
from pathlib import Path

from core.types import GenerationJobParams
from hunyuan_service.config import DEFAULT_FPS, repo_ready
from hunyuan_service.settings import get_hunyuan_settings

logger = logging.getLogger(__name__)


def _resolve_python() -> Path:
    from hunyuan_service.health import _hunyuan_python

    py = _hunyuan_python()
    if py is None:
        raise FileNotFoundError("HUNYUAN_PYTHON not configured or missing")
    return py


def _hunyuan_env(cfg) -> dict[str, str]:
    env = os.environ.copy()
    roots = [str(cfg.project_root), str(cfg.hunyuan_repo_dir)]
    existing = env.get("PYTHONPATH")
    if existing:
        roots.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(roots)
    env["CUDA_VISIBLE_DEVICES"] = cfg.resolved_cuda_device()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["HF_HUB_ENABLE_HF_TRANSFER"] = env.get("HF_HUB_ENABLE_HF_TRANSFER", "0")
    return env


def _video_length(duration_seconds: float | None, fps: int) -> int:
    if duration_seconds is None or duration_seconds <= 0:
        return 121
    return max(25, int(round(float(duration_seconds) * fps)))


def _stream_output(process: subprocess.Popen[str], on_log_line) -> str:
    assert process.stdout is not None
    tail: deque[str] = deque(maxlen=80)

    def _reader() -> None:
        for line in process.stdout:
            s = line.rstrip()
            tail.append(s)
            logger.info("[hunyuan] %s", s)
            if on_log_line:
                on_log_line(s)

    threading.Thread(target=_reader, daemon=True).start()
    process.wait()
    return "\n".join(tail)


def run_hunyuan_generation(params: GenerationJobParams) -> dict:
    cfg = get_hunyuan_settings()
    repo = cfg.hunyuan_repo_dir
    if not repo_ready(repo):
        raise FileNotFoundError(f"Hunyuan generate.py missing under {repo}")
    model_path = cfg.hunyuan_model_path
    if not model_path.is_dir():
        raise FileNotFoundError(f"Hunyuan checkpoints missing: {model_path}")

    mode = (params.mode or params.metadata.get("mode") or "text_to_video").strip().lower()
    if mode not in ("text_to_video", "image_to_video"):
        mode = "text_to_video"
    if mode == "image_to_video":
        if params.image_path is None or not params.image_path.is_file():
            raise FileNotFoundError("Hunyuan image_to_video requires an input image.")

    fps = int(params.fps or cfg.HUNYUAN_DEFAULT_FPS or DEFAULT_FPS)
    video_length = _video_length(params.duration_seconds, fps)
    steps = int(cfg.HUNYUAN_NUM_INFERENCE_STEPS)
    use_step_distill = bool(cfg.HUNYUAN_ENABLE_STEP_DISTILL) and mode == "image_to_video"
    if use_step_distill:
        steps = int(cfg.HUNYUAN_STEP_DISTILL_STEPS)

    py = _resolve_python()
    rewrite = "true" if cfg.HUNYUAN_PROMPT_REWRITE else "false"
    offloading = "true" if cfg.HUNYUAN_ENABLE_OFFLOAD else "false"
    sr = "true" if cfg.HUNYUAN_ENABLE_SR else "false"
    step_distill = "true" if use_step_distill else "false"

    if params.on_log_line:
        params.on_log_line(
            f"[hunyuan_service] {mode} | resolution={cfg.resolved_resolution()} "
            f"frames={video_length} steps={steps} rewrite={rewrite} "
            f"step_distill={step_distill} offload={offloading}"
        )

    cmd: list[str] = [
        str(py),
        "-u",
        "generate.py",
        "--prompt",
        params.prompt,
        "--negative_prompt",
        (params.negative_prompt or "").strip(),
        "--model_path",
        str(model_path),
        "--resolution",
        cfg.resolved_resolution(),
        "--aspect_ratio",
        cfg.resolved_aspect_ratio(),
        "--video_length",
        str(video_length),
        "--num_inference_steps",
        str(steps),
        "--seed",
        str(int(params.seed if params.seed is not None else 123)),
        "--output_path",
        str(params.output_mp4),
        "--rewrite",
        rewrite,
        "--offloading",
        offloading,
        "--sr",
        sr,
        "--enable_step_distill",
        step_distill,
        "--dtype",
        (cfg.HUNYUAN_DTYPE or "bf16").strip(),
    ]
    if mode == "image_to_video":
        cmd.extend(["--image_path", str(params.image_path)])
    else:
        cmd.extend(["--image_path", "none"])

    params.output_mp4.parent.mkdir(parents=True, exist_ok=True)
    tail = _stream_output(
        subprocess.Popen(
            cmd,
            cwd=str(repo),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=_hunyuan_env(cfg),
            start_new_session=True,
        ),
        params.on_log_line,
    )
    if not params.output_mp4.is_file() or params.output_mp4.stat().st_size < 1000:
        raise RuntimeError(
            "Hunyuan did not produce a valid MP4.\n"
            + (tail[-1500:] if tail else "(no subprocess output)")
        )

    return {
        "engine": "hunyuan",
        "engine_used": "hunyuan",
        "mode": mode,
        "video_length": video_length,
        "num_inference_steps": steps,
        "resolution": cfg.resolved_resolution(),
        "duration_seconds": params.duration_seconds,
        "rewrite": cfg.HUNYUAN_PROMPT_REWRITE,
        "step_distill": use_step_distill,
    }