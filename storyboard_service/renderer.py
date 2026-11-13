"""Stage 1 storyboard renderer: smooth pan + push-in on a manga panel (no diffusion)."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from core.config import get_settings


def _ease_in_out(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def render_storyboard_mp4(
    *,
    image_path: Path,
    output_path: Path,
    duration_seconds: float,
    width: int | None = None,
    height: int | None = None,
    fps: int = 24,
    on_log: Callable[[str], None] | None = None,
) -> dict:
    """Render MP4 with combined horizontal pan and slight zoom-in."""
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")

    def log(msg: str) -> None:
        if on_log:
            on_log(msg)

    log("[storyboard] Loading panel image")
    pil = Image.open(image_path).convert("RGB")
    src_w, src_h = pil.size
    frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    out_w = width or src_w
    out_h = height or src_h
    out_w = max(64, int(out_w))
    out_h = max(64, int(out_h))

    total_frames = max(2, int(round(duration_seconds * fps)))
    log(f"[storyboard] Rendering {total_frames} frames at {out_w}x{out_h} @{fps}fps")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    ffmpeg = settings.FFMPEG_BINARY

    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{out_w}x{out_h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None

    pan_fraction = 0.08
    zoom_end = 1.08

    try:
        for i in range(total_frames):
            t = _ease_in_out(i / max(total_frames - 1, 1))
            scale = 1.0 + (zoom_end - 1.0) * t
            crop_w = src_w / scale
            crop_h = src_h / scale
            cx = src_w * (0.5 + pan_fraction * (t - 0.5))
            cy = src_h * 0.5
            x0 = int(max(0, min(src_w - crop_w, cx - crop_w / 2)))
            y0 = int(max(0, min(src_h - crop_h, cy - crop_h / 2)))
            x1 = int(min(src_w, x0 + crop_w))
            y1 = int(min(src_h, y0 + crop_h))
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                crop = frame
            resized = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
            proc.stdin.write(resized.tobytes())
            if i % max(1, total_frames // 10) == 0:
                pct = int(100 * i / total_frames)
                log(f"[storyboard] Encoding frame {i + 1}/{total_frames} ({pct}%)")
    finally:
        proc.stdin.close()

    stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"FFmpeg storyboard encode failed: {stderr[-800:]}")

    log("[storyboard] Finished.")
    return {
        "engine": "storyboard",
        "duration_seconds": duration_seconds,
        "width": out_w,
        "height": out_h,
        "fps": fps,
        "frame_count": total_frames,
        "motion": "pan_zoom_stage1",
    }
