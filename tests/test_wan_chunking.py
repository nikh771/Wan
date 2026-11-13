"""Unit tests for Wan chunk planning (≤5s single-pass, longer multi-chunk)."""

from __future__ import annotations

import pytest

from wan_service.chunking import (
    align_wan_frame_count,
    frames_for_duration,
    plan_wan_chunks,
)


def test_align_wan_frame_count_rule() -> None:
    assert align_wan_frame_count(1) == 5
    assert align_wan_frame_count(120) == 121
    assert align_wan_frame_count(121) == 121
    assert align_wan_frame_count(122) == 125


def test_frames_for_duration_five_seconds() -> None:
    assert frames_for_duration(5.0, 24) == 121


def test_plan_single_pass_at_or_under_five() -> None:
    plan = plan_wan_chunks(5.0, fps=24)
    assert plan.single_pass is True
    assert plan.num_chunks == 1
    assert plan.frames_per_chunk == 121

    short = plan_wan_chunks(3.0, fps=24)
    assert short.single_pass is True
    assert short.num_chunks == 1
    assert short.frames_per_chunk == frames_for_duration(3.0, 24)


def test_plan_multi_chunk_above_five() -> None:
    plan = plan_wan_chunks(12.0, fps=24)
    assert plan.single_pass is False
    assert plan.num_chunks == 3
    assert plan.frames_per_chunk == 121

    plan15 = plan_wan_chunks(15.0, fps=24)
    assert plan15.num_chunks == 3


def test_plan_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        plan_wan_chunks(0)
    with pytest.raises(ValueError):
        plan_wan_chunks(5, fps=0)
