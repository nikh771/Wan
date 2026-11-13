"""Unit tests for LTX frame/resolution helpers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ltx_service.config import (
    align_ltx_frame_count,
    calculate_ltx_frame_count,
    materialize_pipeline_config,
    normalize_ltx_resolution,
)


def test_align_ltx_frame_count_rule() -> None:
    assert align_ltx_frame_count(120) == 121
    assert align_ltx_frame_count(121) == 121
    assert align_ltx_frame_count(8) == 9


def test_calculate_ltx_frame_count_duration() -> None:
    plan = calculate_ltx_frame_count(5.0, 24)
    assert plan.requested_frames == 120
    assert plan.actual_frames == 121
    assert abs(plan.actual_duration_seconds - 5.0416666) < 0.001


def test_normalize_ltx_resolution_divisible_by_32() -> None:
    res = normalize_ltx_resolution(704, 384)
    assert res.width % 32 == 0
    assert res.height % 32 == 0
    assert res.width == 704
    assert res.height == 384


def test_normalize_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        normalize_ltx_resolution(0, 384)


def test_materialize_pipeline_config(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    cfg_dir = repo / "configs"
    cfg_dir.mkdir(parents=True)
    yaml_src = cfg_dir / "ltxv-2b-0.9.6-distilled.yaml"
    yaml_src.write_text(
        "checkpoint_path: relative.safetensors\nguidance_scale: 1\n",
        encoding="utf-8",
    )
    ckpt = tmp_path / "models" / "ltx" / "ltxv-2b-0.9.6-distilled-04-25.safetensors"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"x")
    resolved = materialize_pipeline_config(
        repo,
        "configs/ltxv-2b-0.9.6-distilled.yaml",
        ckpt,
    )
    assert resolved.is_file()
    text = resolved.read_text(encoding="utf-8")
    assert str(ckpt.resolve()) in text


@pytest.mark.skipif(
    os.environ.get("RUN_LTX_INTEGRATION_TEST", "").lower() not in ("1", "true", "yes"),
    reason="Set RUN_LTX_INTEGRATION_TEST=true to run GPU integration test",
)
def test_ltx_integration_smoke() -> None:
    """Optional GPU test — not run in CI by default."""
    from ltx_service.health import ltx_health_payload

    health = ltx_health_payload()
    assert health["engine"] == "ltx"
    assert health["model"] == "ltxv-2b-0.9.6-distilled"
