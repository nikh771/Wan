"""Request / mode validation for CogVideoX engine selection."""

from __future__ import annotations

import pytest

from core.engine_dispatch import normalize_engine
from cogvideox_service.adapter import CogVideoXGenerateRequest, CogVideoXImageToVideoAdapter
from cogvideox_service.config import calculate_cogvideox_frame_count


def test_normalize_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        normalize_engine("hybrid")


def test_t2v_request_defaults() -> None:
    req = CogVideoXGenerateRequest(prompt="x", output_mp4=__import__("pathlib").Path("out.mp4"))
    assert req.mode == "text_to_video"
    assert req.width == 720
    assert req.height == 480
    assert req.fps == 8
    assert req.num_frames == 49
    assert req.steps == 50
    assert req.guidance_scale == 6.0


def test_i2v_adapter_not_implemented() -> None:
    adapter = CogVideoXImageToVideoAdapter()
    caps = adapter.get_capabilities()
    assert caps["implemented"] is False
    with pytest.raises(NotImplementedError):
        adapter.generate()


def test_frame_validation_positive() -> None:
    with pytest.raises(ValueError):
        calculate_cogvideox_frame_count(num_frames=-1, fps=8)
    with pytest.raises(ValueError):
        calculate_cogvideox_frame_count(duration_seconds=0, fps=8)
