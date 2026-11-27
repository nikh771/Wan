"""Wan frame planning (4n+1 at 24 fps output)."""

from __future__ import annotations

from wan_service.chunking import frames_for_duration, plan_wan_chunks


def test_five_seconds_at_24fps_is_121_frames() -> None:
    assert frames_for_duration(5.0, 24) == 121
    plan = plan_wan_chunks(5.0, fps=24, chunk_seconds=10.0)
    assert plan.single_pass
    assert plan.frames_per_chunk == 121
    assert abs(plan.chunk_actual_duration_seconds - 121 / 24) < 0.01