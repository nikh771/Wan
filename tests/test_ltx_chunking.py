"""Unit tests for LTX duration chunk planning."""

from __future__ import annotations

import pytest

from ltx_service.chunking import (
    chunk_duration_seconds,
    chunk_frame_count,
    plan_ltx_chunks,
)


def test_single_pass_under_or_equal_10s() -> None:
    plan = plan_ltx_chunks(5.0, fps=24, chunk_seconds=10.0)
    assert plan.single_pass is True
    assert plan.num_chunks == 1
    assert plan.frames_per_chunk == 121  # 5s * 24 → 120 → 8n+1 = 121


def test_exactly_10s_is_single_pass() -> None:
    plan = plan_ltx_chunks(10.0, fps=24, chunk_seconds=10.0)
    assert plan.single_pass is True
    assert plan.num_chunks == 1
    assert plan.frames_per_chunk == 241  # 240 → 241


def test_over_10s_splits_into_chunks() -> None:
    plan = plan_ltx_chunks(25.0, fps=24, chunk_seconds=10.0)
    assert plan.single_pass is False
    assert plan.num_chunks == 3
    assert plan.frames_per_chunk == 241
    assert plan.last_chunk_frames == 121  # 5s remainder
    assert chunk_frame_count(plan, 0) == 241
    assert chunk_frame_count(plan, 2) == 121
    assert chunk_duration_seconds(plan, 0) == pytest.approx(241 / 24)
    assert chunk_duration_seconds(plan, 2) == pytest.approx(121 / 24)


def test_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        plan_ltx_chunks(0, fps=24)
