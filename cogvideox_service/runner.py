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

from core.chunk_cleanup import cleanup_generation_artifacts
from core.types import GenerationJobParams
from cogvideox_service.chunking import (
    chunk_frame_count,
    plan_cogvideox_chunks,
)
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
from wan_service.utils.ffmpeg import concat_mp4s

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


def _run_cogvideox_worker(
    *,
    mode: str,
    prompt: str,
    negative_prompt: str,
    output_mp4: Path,
    width: int,
    height: int,
    fps: int,
    num_frames: int,
    steps: int,
    guidance: float,
    seed: int,
    memory_mode: str,
    image_path: Path | None,
    on_log_line: Callable[[str], None] | None,
) -> dict:
    cfg = get_cogvideox_settings()
    cmd = [
        cfg.resolved_cogvideox_python(),
        "-u",
        str(_WORKER),
        "--mode",
        mode,
        "--prompt",
        prompt,
        "--negative-prompt",
        negative_prompt or "",
        "--output",
        str(output_mp4.resolve()),
        "--width",
        str(width),
        "--height",
        str(height),
        "--fps",
        str(fps),
        "--num-frames",
        str(num_frames),
        "--steps",
        str(steps),
        "--guidance-scale",
        str(guidance),
        "--seed",
        str(seed),
        "--memory-mode",
        memory_mode,
    ]
    if mode == "image_to_video" and image_path is not None:
        cmd.extend(["--image", str(image_path.resolve())])

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
    tail = _stream_output(proc, on_log_line)
    if proc.returncode != 0:
        raise RuntimeError(f"CogVideoX worker exit {proc.returncode}\n{tail[-2000:]}")
    return _parse_worker_ok(tail)


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
    steps = int(
        params.metadata.get(
            "steps", cfg.COGVIDEOX_DEFAULT_STEPS or DEFAULT_NUM_INFERENCE_STEPS
        )
    )
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

    duration = float(
        params.duration_seconds
        if params.duration_seconds is not None and params.duration_seconds > 0
        else cfg.COGVIDEOX_DEFAULT_DURATION_SECONDS
    )
    duration = max(duration, 0.5)

    # Explicit num_frames overrides duration for single-pass only.
    num_frames_meta = params.metadata.get("num_frames")
    force_single_frames: int | None = None
    if num_frames_meta is not None:
        frame_plan = calculate_cogvideox_frame_count(
            num_frames=int(num_frames_meta), fps=fps
        )
        force_single_frames = frame_plan.actual_frames
        duration = frame_plan.actual_duration_seconds

    res = normalize_cogvideox_resolution(width, height)
    chunk_seconds = float(cfg.COGVIDEOX_CHUNK_SECONDS)
    plan = plan_cogvideox_chunks(
        duration,
        fps=fps,
        chunk_seconds=chunk_seconds,
    )
    work_dir = params.output_mp4.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        if params.on_log_line:
            params.on_log_line(msg)

    if plan.single_pass or force_single_frames is not None:
        frames = force_single_frames or plan.frames_per_chunk
        _log(
            f"[cogvideox_service] Single-pass {mode} | {res.width}x{res.height} "
            f"frames={frames} steps={steps} guidance={guidance} seed={seed}"
        )
        result = _run_cogvideox_worker(
            mode="text_to_video",
            prompt=params.prompt,
            negative_prompt=params.negative_prompt or "",
            output_mp4=params.output_mp4,
            width=res.width,
            height=res.height,
            fps=fps,
            num_frames=frames,
            steps=steps,
            guidance=guidance,
            seed=seed,
            memory_mode=memory_mode,
            image_path=None,
            on_log_line=params.on_log_line,
        )
        result.setdefault("engine", "cogvideox")
        result.setdefault("status", "completed")
        result.setdefault("output_path", str(params.output_mp4))
        result["chunks"] = 1
        result["chunked"] = False
        return result

    _log(
        f"[cogvideox_service] Chunked T2V | chunks={plan.num_chunks} "
        f"× ≤{chunk_seconds:.0f}s (target {duration:.2f}s) | "
        f"{res.width}x{res.height} @ {fps}fps"
    )
    chunk_paths: list[Path] = []

    for i in range(plan.num_chunks):
        frames = chunk_frame_count(plan, i)
        chunk_out = work_dir / f"cog_chunk_{i:02d}.mp4"
        chunk_seed = seed + i
        _log(
            f"[cogvideox_service] Generating chunk {i + 1}/{plan.num_chunks} "
            f"frames={frames} seed={chunk_seed}"
        )
        _run_cogvideox_worker(
            mode="text_to_video",
            prompt=params.prompt,
            negative_prompt=params.negative_prompt or "",
            output_mp4=chunk_out,
            width=res.width,
            height=res.height,
            fps=fps,
            num_frames=frames,
            steps=steps,
            guidance=guidance,
            seed=chunk_seed,
            memory_mode=memory_mode,
            image_path=None,
            on_log_line=params.on_log_line,
        )
        chunk_paths.append(chunk_out)

    _log(
        f"[cogvideox_service] Concatenating {len(chunk_paths)} chunks → "
        f"{params.output_mp4.name}"
    )
    concat_mp4s(
        chunk_paths,
        params.output_mp4,
        skip_first_frame_after_first=True,
        fps=fps,
    )
    removed = cleanup_generation_artifacts(chunk_paths, [])
    _log(f"[cogvideox_service] Removed {len(removed)} intermediate file(s)")

    return {
        "engine": "cogvideox",
        "mode": "text_to_video",
        "chunks": plan.num_chunks,
        "chunked": True,
        "chunk_seconds": plan.chunk_seconds,
        "frames_per_chunk": plan.frames_per_chunk,
        "last_chunk_frames": plan.last_chunk_frames,
        "duration_seconds": duration,
        "width": res.width,
        "height": res.height,
        "fps": fps,
        "seed": seed,
        "status": "completed",
        "output_path": str(params.output_mp4),
        "removed_intermediates": removed,
        "continuity_note": (
            "Sequential T2V chunks concatenated. "
            "Last-frame conditioning across chunks needs a CogVideoX I2V checkpoint."
        ),
    }
