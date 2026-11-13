"""LTX-Video 2B distilled service (isolated from wan_service)."""

from ltx_service.config import (
    LTX_CHECKPOINT_FILENAME,
    LTX_MODEL_ID,
    calculate_ltx_frame_count,
    normalize_ltx_resolution,
)

__all__ = [
    "LTX_MODEL_ID",
    "LTX_CHECKPOINT_FILENAME",
    "calculate_ltx_frame_count",
    "normalize_ltx_resolution",
]
