"""CogVideoX-2B service (isolated from wan_service / ltx_service)."""

from cogvideox_service.config import (
    COGVIDEOX_HF_ID,
    COGVIDEOX_MODEL_ID,
    calculate_cogvideox_duration,
    calculate_cogvideox_frame_count,
    normalize_cogvideox_resolution,
)

__all__ = [
    "COGVIDEOX_HF_ID",
    "COGVIDEOX_MODEL_ID",
    "calculate_cogvideox_duration",
    "calculate_cogvideox_frame_count",
    "normalize_cogvideox_resolution",
]
