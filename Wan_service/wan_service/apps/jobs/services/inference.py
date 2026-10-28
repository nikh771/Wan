"""Run official Wan2.2 generate.py for TI2V-5B image-to-video."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from PIL import Image

from wan_service.core.config import get_settings, settings

logger = logging.getLogger(__name__)


def select_wan_size(image_path: Path) -> str:
    with Image.open(image_path) as im:
        w, h = im.size
    if w >= h:
        return settings.WAN_SIZE_LANDSCAPE
    return settings.WAN_SIZE_PORTRAIT


def run_wan_inference(
    *,
    image_path: Path,
    prompt: str,
    output_mp4: Path,
) -> None:
    script = settings.wan_generate_script
    ckpt = settings.wan_ckpt_dir
    repo = settings.wan_repo_dir

    if not script.exists():
        raise FileNotFoundError(f"Wan generate.py not found: {script}")
    if not ckpt.exists():
        raise FileNotFoundError(f"Wan checkpoint dir not found: {ckpt}")

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    size = select_wan_size(image_path)

    cmd = [
        settings.WAN_PYTHON,
        str(script),
        "--task",
        "ti2v-5B",
        "--size",
        size,
        "--ckpt_dir",
        str(ckpt),
        "--offload_model",
        "True",
        "--convert_model_dtype",
        "--t5_cpu",
        "--image",
        str(image_path),
        "--prompt",
        prompt,
        "--save_file",
        str(output_mp4),
    ]

    env = os.environ.copy()
    env["WAN_INFERENCE_BACKEND"] = get_settings().resolved_inference_backend()

    logger.info(
        "Starting Wan inference cwd=%s size=%s backend=%s",
        repo,
        size,
        env["WAN_INFERENCE_BACKEND"],
    )
    result = subprocess.run(
        cmd,
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=None,
        env=env,
    )
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "")[-2000:]
        raise RuntimeError(f"Wan generate.py failed (exit {result.returncode}): {tail}")

    if not output_mp4.exists() or output_mp4.stat().st_size < 1000:
        raise RuntimeError("Wan inference did not produce a valid MP4 file.")
