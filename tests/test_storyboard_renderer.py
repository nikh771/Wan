"""Storyboard MP4 output tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from PIL import Image

from wan_service.apps.jobs.storyboard_renderer import render_storyboard_mp4
from wan_service.utils.ffmpeg import validate_output_mp4


@pytest.fixture
def panel_png(tmp_path: Path) -> Path:
    path = tmp_path / "panel.png"
    Image.new("RGB", (320, 180), color=(40, 80, 120)).save(path)
    return path


def test_storyboard_renders_valid_mp4(panel_png: Path, tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    out = tmp_path / "out.mp4"
    meta = render_storyboard_mp4(
        image_path=panel_png,
        output_path=out,
        duration_seconds=0.5,
        width=320,
        height=180,
        fps=12,
    )
    assert out.is_file()
    validate_output_mp4(out)
    assert meta["engine"] == "storyboard"
    assert meta["frame_count"] >= 2
