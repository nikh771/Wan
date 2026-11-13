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


def extract_last_frame(source: Path, dest: Path) -> None:
    """Write the last video frame of ``source`` as a PNG at ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        settings.FFMPEG_BINARY,
        "-y",
        "-sseof",
        "-0.05",
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(dest),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not dest.exists() or dest.stat().st_size < 100:
        # Fallback: seek near end via duration probe
        info = probe_video(source)
        duration = float((info.get("format") or {}).get("duration") or 0)
        seek = max(0.0, duration - 0.05)
        cmd = [
            settings.FFMPEG_BINARY,
            "-y",
            "-ss",
            f"{seek:.3f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(dest),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0 or not dest.exists() or dest.stat().st_size < 100:
            raise RuntimeError(
                f"FFmpeg extract last frame failed: {(result.stderr or '')[-500:]}"
            )


def concat_mp4s(
    sources: list[Path],
    dest: Path,
    *,
    skip_first_frame_after_first: bool = True,
    fps: int = 24,
) -> None:
    """Concatenate MP4s; optionally drop frame 0 of chunks after the first (I2V bridge)."""
    if not sources:
        raise ValueError("concat_mp4s requires at least one source")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if len(sources) == 1:
        import shutil

        shutil.copy2(sources[0], dest)
        return

    if not skip_first_frame_after_first:
        list_file = dest.parent / f"{dest.stem}_concat.txt"
        lines = [f"file '{path.resolve()}'" for path in sources]
        list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        cmd = [
            settings.FFMPEG_BINARY,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            str(dest),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            cmd = [
                settings.FFMPEG_BINARY,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(dest),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg concat failed: {(result.stderr or '')[-500:]}")
        return

    # filter_complex: keep all of chunk0; from chunk1.. skip first frame
    inputs: list[str] = []
    for path in sources:
        inputs.extend(["-i", str(path)])
    n = len(sources)
    parts: list[str] = ["[0:v]"]
    filter_steps: list[str] = []
    for i in range(1, n):
        filter_steps.append(
            f"[{i}:v]trim=start_frame=1,setpts=PTS-STARTPTS[v{i}]"
        )
        parts.append(f"[v{i}]")
    filter_steps.append(f"{''.join(parts)}concat=n={n}:v=1:a=0[outv]")
    cmd = [
        settings.FFMPEG_BINARY,
        "-y",
        *inputs,
        "-filter_complex",
        ";".join(filter_steps),
        "-map",
        "[outv]",
        "-r",
        str(max(1, fps)),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(dest),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg filter concat failed: {(result.stderr or '')[-500:]}")
