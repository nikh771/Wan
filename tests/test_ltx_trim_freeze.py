"""Unit tests for trailing freeze detection."""

from __future__ import annotations

import numpy as np

from ltx_service.trim_freeze import find_trailing_freeze_cut


def test_find_trailing_freeze_cut_trims_static_tail() -> None:
    # 40 moving frames + 15 static copies of the last moving frame
    frames = []
    for i in range(40):
        frames.append(np.full((8, 8), float(i), dtype=np.float32))
    last = frames[-1]
    for _ in range(15):
        frames.append(last.copy())
    keep = find_trailing_freeze_cut(
        frames,
        diff_threshold=2.0,
        min_keep_ratio=0.5,
        min_trim_frames=0,
        max_trim_frames=30,
    )
    assert keep <= 42  # motion part kept, freeze mostly gone
    assert keep >= 20


def test_min_trim_frames_enforced() -> None:
    frames = [np.full((4, 4), float(i), dtype=np.float32) for i in range(50)]
    keep = find_trailing_freeze_cut(
        frames,
        diff_threshold=0.0,  # no freeze detected by diff
        min_keep_ratio=0.5,
        min_trim_frames=12,
        max_trim_frames=40,
    )
    assert keep == 38  # 50 - 12
