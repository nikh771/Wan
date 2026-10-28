"""FFmpeg / ffprobe helpers for post-processing generated video."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from wan_service.core.config import settings


def probe_video(path: Path) -> dict:
    cmd = [
        settings.FFPROBE_BINARY,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError("ffprobe could not read generated video.")
    return json.loads(result.stdout or "{}")


def trim_video_to_duration(source: Path, dest: Path, duration_seconds: float) -> None:
    if duration_seconds <= 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        settings.FFMPEG_BINARY,
        "-y",
        "-i",
        str(source),
        "-t",
        str(duration_seconds),
        "-c",
        "copy",
        str(dest),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        cmd = [
            settings.FFMPEG_BINARY,
            "-y",
            "-i",
            str(source),
            "-t",
            str(duration_seconds),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(dest),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg trim failed: {result.stderr[-500:]}")


def validate_output_mp4(path: Path) -> None:
    probe_video(path)
    if path.stat().st_size < 1000:
        raise RuntimeError("Generated MP4 is empty or too small.")
