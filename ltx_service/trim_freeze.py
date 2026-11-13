"""Trim trailing near-static frames from LTX chunk MP4s.

Distilled LTX often settles onto a held last frame at the end of a clip. That
freeze is then used as the next-chunk bridge and also appears in the final
concat, which makes the video look stuck.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def _write_frames_mp4(frames, path: Path, fps: float, width: int, height: int) -> None:
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    raw_tmp = path.with_suffix(".trim_raw.mp4")
    writer = cv2.VideoWriter(str(raw_tmp), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {raw_tmp}")
    for frame in frames:
        writer.write(frame)
    writer.release()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        h264_tmp = path.with_suffix(".trim_h264.mp4")
        proc = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(raw_tmp),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
                "-preset",
                "fast",
                "-an",
                str(h264_tmp),
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0 and h264_tmp.is_file():
            raw_tmp.unlink(missing_ok=True)
            h264_tmp.replace(path)
            return
        h264_tmp.unlink(missing_ok=True)
        logger.warning("ffmpeg remux failed after trim; keeping mp4v (%s)", (proc.stderr or "")[-200:])

    raw_tmp.replace(path)


def find_trailing_freeze_cut(
    gray_frames: list,
    *,
    diff_threshold: float = 2.0,
    min_keep_ratio: float = 0.75,
    min_trim_frames: int = 0,
    max_trim_frames: int | None = None,
) -> int:
    """Return keep count (number of frames to keep from the start)."""
    import numpy as np

    n = len(gray_frames)
    if n < 8:
        return n

    diffs = [
        float(np.mean(np.abs(gray_frames[i] - gray_frames[i - 1])))
        for i in range(1, n)
    ]
    # Walk backward while incoming frame-to-frame diff is below threshold.
    cut_idx = n - 1  # last kept frame index
    for i in range(len(diffs) - 1, -1, -1):
        if diffs[i] < diff_threshold:
            cut_idx = i  # keep frames[0:i+1] → drop from i+1 onward
        else:
            break

    keep = cut_idx + 1
    # Always drop at least min_trim_frames from the end when requested (chunk seams).
    if min_trim_frames > 0:
        keep = min(keep, n - min_trim_frames)

    min_keep = max(8, int(n * min_keep_ratio))
    keep = max(keep, min_keep)

    if max_trim_frames is not None:
        keep = max(keep, n - max_trim_frames)

    keep = min(keep, n)
    return keep


def trim_trailing_freeze_mp4(
    mp4_path: Path,
    *,
    diff_threshold: float = 2.0,
    min_keep_ratio: float = 0.75,
    min_trim_frames: int = 0,
    max_trim_frames: int = 48,
) -> int:
    """Rewrite ``mp4_path`` without trailing near-static frames. Returns frames removed."""
    import cv2
    import numpy as np

    mp4_path = Path(mp4_path)
    if not mp4_path.is_file():
        raise FileNotFoundError(mp4_path)

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {mp4_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if len(frames) < 8:
        return 0

    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    keep = find_trailing_freeze_cut(
        gray,
        diff_threshold=diff_threshold,
        min_keep_ratio=min_keep_ratio,
        min_trim_frames=min_trim_frames,
        max_trim_frames=max_trim_frames,
    )
    removed = len(frames) - keep
    if removed <= 0:
        return 0

    backup = mp4_path.with_suffix(".pre_trim.mp4")
    if not backup.exists():
        shutil.copy2(mp4_path, backup)

    _write_frames_mp4(frames[:keep], mp4_path, fps, width, height)
    logger.info(
        "Trimmed trailing freeze from %s: removed=%s keep=%s/%s",
        mp4_path.name,
        removed,
        keep,
        len(frames),
    )
    return removed
