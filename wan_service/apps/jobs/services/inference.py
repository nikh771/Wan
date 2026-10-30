"""Run official Wan2.2 generate.py for TI2V-5B image-to-video."""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from collections import deque
from pathlib import Path

from PIL import Image

from wan_service.core.config import get_settings, settings
from wan_service.core.inference_deps import torch_optional_probe_cuda

logger = logging.getLogger(__name__)

_LOG_TAIL_LINES = 40


def select_wan_size(image_path: Path) -> str:
    with Image.open(image_path) as im:
        w, h = im.size
    if w >= h:
        return settings.WAN_SIZE_LANDSCAPE
    return settings.WAN_SIZE_PORTRAIT


def resolve_gpu_only_inference() -> tuple[bool, str]:
    """Return (use_full_gpu, reason) for generate.py flags."""
    cfg = get_settings()
    if cfg.resolved_inference_backend() != "linux_cuda":
        return False, "backend is not linux_cuda"
    if not cfg.WAN_INFERENCE_GPU_ONLY:
        return False, "WAN_INFERENCE_GPU_ONLY is disabled"

    _available, _detail, total_gb, _vram_ok = torch_optional_probe_cuda()
    if total_gb is None:
        return False, "could not read GPU VRAM; using memory-saving mode"
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
    env["CUDA_VISIBLE_DEVICES"] = cfg.WAN_CUDA_DEVICE.strip() or "0"
    prefix = str(repo)
    env["PYTHONPATH"] = prefix + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if not gpu_only and not env.get("PYTORCH_CUDA_ALLOC_CONF"):
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    for key in list(env.keys()):
        if key.startswith(("DEBUGPY", "PYDEVD", "VSCODE_DEBUG", "BUNDLED_DEBUGPY")):
            env.pop(key, None)
    env.pop("PYTHONDEBUG", None)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _stream_subprocess_output(process: subprocess.Popen[str]) -> deque[str]:
    assert process.stdout is not None
    tail: deque[str] = deque(maxlen=_LOG_TAIL_LINES)

    def _reader() -> None:
        for line in process.stdout:
            stripped = line.rstrip()
            tail.append(stripped)
            logger.info("[wan generate] %s", stripped)

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    process.wait()
    thread.join(timeout=1)
    return tail


def run_wan_inference(
    *,
    image_path: Path,
    prompt: str,
    output_mp4: Path,
) -> None:
    cfg = get_settings()
    script = cfg.wan_generate_script
    ckpt = cfg.wan_ckpt_dir
    repo = cfg.wan_repo_dir

    if not script.exists():
        raise FileNotFoundError(f"Wan generate.py not found: {script}")
    if not ckpt.exists():
        raise FileNotFoundError(f"Wan checkpoint dir not found: {ckpt}")

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    size = select_wan_size(image_path)
    gpu_only, gpu_mode_reason = resolve_gpu_only_inference()

    cmd: list[str] = [
        cfg.WAN_PYTHON,
        str(script),
        "--task",
        "ti2v-5B",
        "--size",
        size,
        "--ckpt_dir",
        str(ckpt),
        "--offload_model",
        "False" if gpu_only else "True",
        "--convert_model_dtype",
        "--init_on_cpu",
        "False" if gpu_only else "True",
        "--image",
        str(image_path),
        "--prompt",
        prompt,
        "--save_file",
        str(output_mp4),
    ]
    if not gpu_only:
        cmd.append("--t5_cpu")

    env = _inference_env(repo=repo, gpu_only=gpu_only)

    logger.info(
        "Starting Wan inference cwd=%s size=%s backend=%s gpu_only=%s (%s) cuda_device=%s",
        repo,
        size,
        env["WAN_INFERENCE_BACKEND"],
        gpu_only,
        gpu_mode_reason,
        env["CUDA_VISIBLE_DEVICES"],
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
    log_tail = _stream_subprocess_output(process)

    if process.returncode != 0:
        detail = "\n".join(log_tail) if log_tail else "(no subprocess output captured)"
        raise RuntimeError(
            f"Wan generate.py failed (exit {process.returncode}). Last output:\n{detail}"
        )

    if not output_mp4.exists() or output_mp4.stat().st_size < 1000:
        raise RuntimeError("Wan inference did not produce a valid MP4 file.")
