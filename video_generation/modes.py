"""Video mode normalization (t2v / i2v)."""

from __future__ import annotations

from core.types import VideoMode

_MODE_ALIASES: dict[str, VideoMode] = {
    "t2v": "text_to_video",
    "text_to_video": "text_to_video",
    "text-to-video": "text_to_video",
    "i2v": "image_to_video",
    "image_to_video": "image_to_video",
    "image-to-video": "image_to_video",
}


def video_mode_label(mode: VideoMode) -> str:
    return "t2v" if mode == "text_to_video" else "i2v"


def normalize_video_mode(value: str) -> VideoMode:
    key = (value or "").strip().lower()
    if key not in _MODE_ALIASES:
        raise ValueError(
            f"Unsupported mode {value!r}. Use t2v, i2v, text_to_video, or image_to_video."
        )
    return _MODE_ALIASES[key]
