"""Run official Wan2.2 generate.py for TI2V-5B text/image-to-video."""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from collections import deque
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from wan_service.core.config import get_settings, settings
from wan_service.core.inference_deps import torch_optional_probe_cuda

logger = logging.getLogger(__name__)

_LOG_TAIL_LINES = 40


def select_wan_size(image_path: Path | None, *, width: int | None = None, height: int | None = None) -> str:
    if image_path is not None and image_path.is_file():
        with Image.open(image_path) as im:
            w, h = im.size
        if w >= h:
            return settings.WAN_SIZE_LANDSCAPE
        return settings.WAN_SIZE_PORTRAIT
    if width is not None and height is not None and height > width:
        return settings.WAN_SIZE_PORTRAIT
    return settings.WAN_SIZE_LANDSCAPE


def resolve_gpu_only_inference() -> tuple[bool, str]:
    """Return (use_full_gpu, reason) for generate.py flags."""
    cfg = get_settings()
    if cfg.resolved_inference_backend() != "linux_cuda":
        return False, "backend is not linux_cuda"
    if not cfg.WAN_INFERENCE_GPU_ONLY:
        return False, "WAN_INFERENCE_GPU_ONLY is disabled"
    if cfg.WAN_INFERENCE_FORCE_GPU_ONLY:
        return True, "WAN_INFERENCE_FORCE_GPU_ONLY enabled (full GPU, no VRAM gate)"

    _available, _detail, total_gb, _vram_ok = torch_optional_probe_cuda()
    if total_gb is None:
        return (
            True,
            "WAN_INFERENCE_GPU_ONLY enabled (VRAM not probed in API process; "
            "subprocess uses full-GPU generate.py flags)",
        )
    if total_gb < cfg.WAN_GPU_ONLY_MIN_VRAM_GB:
        return (
            False,
            f"GPU VRAM {total_gb:.1f} GB < WAN_GPU_ONLY_MIN_VRAM_GB "
            f"({cfg.WAN_GPU_ONLY_MIN_VRAM_GB:.1f} GB); using T5-on-CPU + offload",
        )
    return True, f"GPU VRAM {total_gb:.1f} GB sufficient for full-GPU mode"


def _inference_env(*, repo: Path, gpu_only: bool) -> dict[str, str]:
    cfg = get_settings()
    env = os.environ.copy()
    env["WAN_INFERENCE_BACKEND"] = cfg.resolved_inference_backend()
    if cfg.WAN_INFERENCE_GPU_ONLY:
        env["WAN_INFERENCE_GPU_ONLY"] = "true"
    if cfg.WAN_INFERENCE_FORCE_GPU_ONLY:
        env["WAN_INFERENCE_FORCE_GPU_ONLY"] = "true"
    env["CUDA_VISIBLE_DEVICES"] = cfg.WAN_CUDA_DEVICE.strip() or "0"
    roots = [str(cfg.project_root), str(repo)]
    existing = env.get("PYTHONPATH")
    if existing:
        roots.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(roots)
    if not gpu_only and not env.get("PYTORCH_CUDA_ALLOC_CONF"):
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    for key in list(env.keys()):
        if key.startswith(("DEBUGPY", "PYDEVD", "VSCODE_DEBUG", "BUNDLED_DEBUGPY")):
            env.pop(key, None)
    env.pop("PYTHONDEBUG", None)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _stream_subprocess_output(
    process: subprocess.Popen[str],
    *,
    on_log_line: Callable[[str], None] | None = None,
) -> deque[str]:
    assert process.stdout is not None
    tail: deque[str] = deque(maxlen=_LOG_TAIL_LINES)

    def _reader() -> None:
        for line in process.stdout:
            stripped = line.rstrip()
            tail.append(stripped)
            logger.info("[wan generate] %s", stripped)
            if on_log_line is not None:
                on_log_line(stripped)

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    process.wait()
    thread.join(timeout=1)
    return tail


def _resolve_frame_num(override: int | None) -> int | None:
    cfg = get_settings()
    frame_num = override if override is not None else cfg.WAN_INFERENCE_FRAME_NUM
    if frame_num is not None and (frame_num <= 0 or (frame_num - 1) % 4 != 0):
        raise ValueError(
            f"frame_num must be 4n+1 (got {frame_num}); e.g. 49, 81, 121"
        )
    return frame_num


def run_wan_inference(
    *,
    job_id: str = "local",
    image_path: Path | None,
    prompt: str,
    output_mp4: Path,
    on_log_line: Callable[[str], None] | None = None,
    frame_num: int | None = None,
    seed: int | None = None,
    width: int | None = None,
    height: int | None = None,
    size: str | None = None,
) -> None:
    cfg = get_settings()
    script = cfg.wan_generate_script
    ckpt = cfg.wan_ckpt_dir
    repo = cfg.wan_repo_dir

    if not script.exists():
        raise FileNotFoundError(f"Wan generate.py not found: {script}")
    if not ckpt.exists():
        raise FileNotFoundError(f"Wan checkpoint dir not found: {ckpt}")
    if image_path is not None and not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    size_key = size or select_wan_size(image_path, width=width, height=height)
    gpu_only, gpu_mode_reason = resolve_gpu_only_inference()
    resolved_frames = _resolve_frame_num(frame_num)

    def _run_persistent() -> None:
        from wan_service.apps.jobs.services.persistent_worker import get_persistent_worker

        worker = get_persistent_worker()
        worker.run_job(
            job_id=job_id,
            image_path=image_path,
            prompt=prompt,
            output_mp4=output_mp4,
            size=size_key,
            sample_steps=cfg.WAN_INFERENCE_SAMPLE_STEPS,
            frame_num=resolved_frames,
            seed=seed,
            on_log_line=on_log_line,
        )

    if cfg.WAN_PERSISTENT_INFERENCE and cfg.resolved_inference_backend() == "linux_cuda":
        from wan_service.apps.jobs.services.persistent_worker import get_persistent_worker

        worker = get_persistent_worker()
        if worker.is_ready:
            logger.info(
                "Using persistent inference worker for job %s (gpu_only=%s, %s)",
                job_id,
                gpu_only,
                gpu_mode_reason,
            )
            _run_persistent()
            return
        if worker.is_loading:
            logger.info(
                "Persistent worker still loading; waiting for job %s (no generate.py fallback)",
                job_id,
            )
            if on_log_line is not None:
                on_log_line(
                    "[wan_service] Waiting for warm worker to finish loading (one model on GPU)…"
                )
            worker.wait_until_ready(timeout_sec=cfg.WAN_WORKER_LOAD_TIMEOUT_SEC)
            _run_persistent()
            return
        if worker.init_error:
            logger.warning(
                "Persistent worker unavailable (%s); falling back to generate.py",
                worker.init_error,
            )

    cmd: list[str] = [
        cfg.WAN_PYTHON,
        str(script),
        "--task",
        "ti2v-5B",
        "--size",
        size_key,
        "--ckpt_dir",
        str(ckpt),
        "--offload_model",
        "False" if gpu_only else "True",
        "--convert_model_dtype",
        "--init_on_cpu",
        "False" if gpu_only else "True",
        "--prompt",
        prompt,
        "--save_file",
        str(output_mp4),
    ]
    if image_path is not None:
        cmd.extend(["--image", str(image_path)])
    if not gpu_only:
        cmd.append("--t5_cpu")

    if cfg.WAN_INFERENCE_SAMPLE_STEPS is not None:
        cmd.extend(["--sample_steps", str(cfg.WAN_INFERENCE_SAMPLE_STEPS)])
    if resolved_frames is not None:
        cmd.extend(["--frame_num", str(resolved_frames)])
    if seed is not None:
        cmd.extend(["--base_seed", str(seed)])

    env = _inference_env(repo=repo, gpu_only=gpu_only)

    logger.info(
        "Starting Wan inference cwd=%s size=%s steps=%s frames=%s image=%s backend=%s "
        "gpu_only=%s (%s) cuda_device=%s",
        repo,
        size_key,
        cfg.WAN_INFERENCE_SAMPLE_STEPS if cfg.WAN_INFERENCE_SAMPLE_STEPS is not None else "default",
        resolved_frames if resolved_frames is not None else "default",
        image_path is not None,
        env["WAN_INFERENCE_BACKEND"],
        gpu_only,
        gpu_mode_reason,
        env["CUDA_VISIBLE_DEVICES"],
    )
    if on_log_line is not None:
        mode = "i2v" if image_path is not None else "t2v"
        on_log_line(
            f"[wan_service] Spawning generate.py ({mode}, gpu_only={gpu_only}, size={size_key})"
        )
    process = subprocess.Popen(
        cmd,
        cwd=str(repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        start_new_session=True,
    )
    log_tail = _stream_subprocess_output(process, on_log_line=on_log_line)

    if process.returncode != 0:
        detail = "\n".join(log_tail) if log_tail else "(no subprocess output captured)"
        raise RuntimeError(
            f"Wan generate.py failed (exit {process.returncode}). Last output:\n{detail}"
        )

    if not output_mp4.exists() or output_mp4.stat().st_size < 1000:
        raise RuntimeError("Wan inference did not produce a valid MP4 file.")
