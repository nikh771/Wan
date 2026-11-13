"""Amplify visible I2V motion by exaggerating deviation from the first frame.

Distilled LTX often produces low per-frame deltas that look nearly still. Scaling
(frame - first) makes the same latent path read as clearer motion without a
heavier model.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def amplify_mp4_motion(
    mp4_path: Path,
    *,
    scale: float = 2.2,
    keep_original_suffix: str = ".pre_boost.mp4",
) -> Path:
    """Rewrite mp4 in place with amplified motion. Returns output path."""
    import cv2
    import numpy as np

    mp4_path = Path(mp4_path)
    if scale <= 1.01:
        return mp4_path
    if not mp4_path.is_file():
        raise FileNotFoundError(mp4_path)

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {mp4_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames: list = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame.astype(np.float32))
    cap.release()
    if len(frames) < 3:
        return mp4_path

    first = frames[0]
    boosted = [
        np.clip(first + scale * (frame - first), 0, 255).astype(np.uint8)
        for frame in frames
    ]

    backup = mp4_path.with_suffix(keep_original_suffix)
    if not backup.exists():
        shutil.copy2(mp4_path, backup)

    raw_tmp = mp4_path.with_suffix(".boost_raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_tmp), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {raw_tmp}")
    for frame in boosted:
        writer.write(frame)
    writer.release()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        h264_tmp = mp4_path.with_suffix(".boost_h264.mp4")
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
        raw_tmp.unlink(missing_ok=True)
        if proc.returncode == 0 and h264_tmp.is_file():
            h264_tmp.replace(mp4_path)
        else:
            logger.warning("ffmpeg remux failed (%s)", (proc.stderr or "")[-400:])
            # Last resort: write mp4v directly to destination.
            writer = cv2.VideoWriter(str(mp4_path), fourcc, float(fps), (width, height))
            for frame in boosted:
                writer.write(frame)
            writer.release()
    else:
        raw_tmp.replace(mp4_path)

    logger.info(
        "Amplified LTX motion scale=%.2f frames=%s backup=%s",
        scale,
        len(boosted),
        backup.name,
    )
    return mp4_path
