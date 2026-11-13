"""Unit tests for CogVideoX config helpers (no model download)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from cogvideox_service.config import (
    LICENSE_METADATA,
    align_cogvideox_frame_count,
    calculate_cogvideox_duration,
    calculate_cogvideox_frame_count,
    model_available,
    normalize_cogvideox_resolution,
)
from cogvideox_service.gpu_state import GpuState, read_gpu_state, write_gpu_state
from core.engine_dispatch import normalize_engine


def test_align_cogvideox_frame_count_rule() -> None:
    assert align_cogvideox_frame_count(49) == 49
    assert align_cogvideox_frame_count(48) == 49
    assert align_cogvideox_frame_count(8) == 9
    assert align_cogvideox_frame_count(50) == 57


def test_calculate_cogvideox_frame_count_default() -> None:
    plan = calculate_cogvideox_frame_count(num_frames=49, fps=8)
    assert plan.actual_frames == 49
    assert abs(plan.actual_duration_seconds - 6.125) < 1e-6


def test_calculate_from_duration() -> None:
    plan = calculate_cogvideox_frame_count(duration_seconds=6.125, fps=8)
    assert plan.requested_frames == 49
    assert plan.actual_frames == 49


def test_duration_helper() -> None:
    assert abs(calculate_cogvideox_duration(49, 8) - 6.125) < 1e-9
    with pytest.raises(ValueError):
        calculate_cogvideox_duration(0, 8)


def test_normalize_resolution_divisible_by_16() -> None:
    res = normalize_cogvideox_resolution(720, 480)
    assert res.width == 720
    assert res.height == 480
    assert res.width % 16 == 0
    assert res.height % 16 == 0


def test_normalize_rounds_up() -> None:
    res = normalize_cogvideox_resolution(721, 481)
    assert res.width == 736
    assert res.height == 496


def test_normalize_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        normalize_cogvideox_resolution(0, 480)


def test_model_available_false_for_empty(tmp_path: Path) -> None:
    assert model_available(tmp_path) is False
    (tmp_path / "model_index.json").write_text("{}", encoding="utf-8")
    assert model_available(tmp_path) is False


def test_license_metadata() -> None:
    assert LICENSE_METADATA["model_license"] == "Apache-2.0"
    assert LICENSE_METADATA["commercial_use"] is True
    assert "2B" in LICENSE_METADATA["model"] or "2b" in LICENSE_METADATA["hf_id"]


def test_engine_registry_includes_cogvideox() -> None:
    assert normalize_engine("cogvideox") == "cogvideox"
    assert normalize_engine("COGVIDEOX") == "cogvideox"


def test_gpu_state_transitions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WAN_JOBS_DIR", str(tmp_path))
    write_gpu_state(GpuState.IDLE)
    assert read_gpu_state()["state"] == "IDLE"
    write_gpu_state(GpuState.COGVIDEOX_LOADED, owner="cogvideox")
    assert read_gpu_state()["state"] == "COGVIDEOX_LOADED"
    write_gpu_state(GpuState.GENERATING, owner="cogvideox")
    assert read_gpu_state()["state"] == "GENERATING"
    write_gpu_state(GpuState.IDLE)
    assert read_gpu_state()["state"] == "IDLE"


def test_health_payload_shape() -> None:
    from cogvideox_service.health import cogvideox_health_payload

    payload = cogvideox_health_payload()
    assert payload["engine"] == "cogvideox"
    assert payload["model"] == "CogVideoX-2b"
    assert "loaded" in payload
    assert "cuda_available" in payload
    assert "status" in payload
    assert payload["status"] in ("ready", "unavailable")


@pytest.mark.skipif(
    os.environ.get("RUN_COGVIDEOX_INTEGRATION_TEST", "").lower() not in ("1", "true", "yes"),
    reason="Set RUN_COGVIDEOX_INTEGRATION_TEST=true to run GPU integration test",
)
def test_cogvideox_integration_smoke() -> None:
    from cogvideox_service.health import cogvideox_health_payload

    health = cogvideox_health_payload()
    assert health["engine"] == "cogvideox"
    assert health["status"] == "ready"
    assert health["cuda_available"] is True
