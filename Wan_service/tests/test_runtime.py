from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from wan_service.core.config import Settings, clear_settings_cache, get_settings
from wan_service.core.runtime import (
    CudaProbe,
    InferenceReadiness,
    MpsProbe,
    evaluate_inference_readiness,
    health_payload,
)


def test_invalid_runtime_mode_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WAN_RUNTIME_MODE", "windows_cuda")
    clear_settings_cache()
    with pytest.raises(ValidationError, match="Unsupported WAN_RUNTIME_MODE"):
        Settings()


def test_linux_cuda_mode_reports_not_ready_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WAN_RUNTIME_MODE", "linux_cuda")
    clear_settings_cache()
    state = evaluate_inference_readiness()
    assert state.runtime_mode == "linux_cuda"
    assert state.ready is False
    assert state.cuda.available is False or not state.checkpoint_ready


def test_health_cuda_and_mps_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WAN_RUNTIME_MODE", "mac_dev")
    clear_settings_cache()
    body = health_payload()
    assert "cuda_available" in body
    assert "mps_available" in body
    assert body["gpu_available"] is body["cuda_available"]
    assert body["api_available"] is True
    assert body["inference_ready"] is False
    assert body.get("inference_unavailable_reason")
    assert body["inference_dependencies_ready"] is None
    assert "ffmpeg_ready" in body
    reason = body["inference_unavailable_reason"] or ""
    assert str(get_settings().wan_ckpt_dir) not in reason


def test_missing_generate_script_reflected_in_health(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("WAN_RUNTIME_MODE", "linux_cuda")
    missing = tmp_path / "nope.py"
    monkeypatch.setenv("WAN_GENERATE_SCRIPT", str(missing))
    clear_settings_cache()
    body = health_payload()
    assert body["generate_script_ready"] is False
    assert body["inference_ready"] is False


def test_linux_cuda_missing_checkpoint_blocks_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_ready = InferenceReadiness(
        ready=False,
        detail="checkpoint missing",
        public_reason="Model checkpoint directory is missing or empty.",
        runtime_mode="linux_cuda",
        checkpoint_ready=False,
        generate_script_ready=True,
        ffmpeg_ready=True,
        wan_python_ready=True,
        inference_dependencies_ready=True,
        cuda=CudaProbe(available=True, detail="gpu", vram_gb=24.0, vram_sufficient=True),
        mps=MpsProbe(available=False, detail="n/a"),
    )

    monkeypatch.setattr(
        "wan_service.core.runtime.evaluate_inference_readiness",
        lambda: fake_ready,
    )
    body = health_payload()
    assert body["checkpoint_ready"] is False
    assert body["inference_ready"] is False


def test_linux_cuda_missing_dependencies_blocks_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr(
        "wan_service.core.runtime.generate_script_ready",
        lambda: True,
    )
    monkeypatch.setattr(
        "wan_service.core.runtime.probe_wan_python",
        lambda: (True, sys.executable),
    )
    monkeypatch.setattr(
        "wan_service.core.runtime.probe_inference_dependencies",
        lambda: (False, "Wan inference packages missing in WAN_PYTHON (install requirements-linux-cuda.txt)."),
    )
    monkeypatch.setenv("WAN_RUNTIME_MODE", "linux_cuda")
    clear_settings_cache()
    state = evaluate_inference_readiness()
    assert state.ready is False
    assert state.inference_dependencies_ready is False
    assert "requirements-linux-cuda" in state.public_reason


def test_linux_cuda_unavailable_cuda_blocks_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setenv("WAN_RUNTIME_MODE", "linux_cuda")
    clear_settings_cache()
    monkeypatch.setattr(
        "wan_service.core.runtime.torch_optional_probe_cuda",
        lambda: (False, "CUDA not available", None, False),
    )
    monkeypatch.setattr(
        "wan_service.core.runtime.probe_inference_dependencies",
        lambda: (True, "ok"),
    )
    monkeypatch.setattr(
        "wan_service.core.runtime.probe_wan_python",
        lambda: (True, sys.executable),
    )
    monkeypatch.setattr(
        "wan_service.core.runtime.generate_script_ready",
        lambda: True,
    )
    monkeypatch.setattr(
        "wan_service.core.runtime.checkpoint_ready",
        lambda: True,
    )
    monkeypatch.setattr(
        "wan_service.core.runtime.probe_ffmpeg",
        lambda: True,
    )
    state = evaluate_inference_readiness()
    assert state.ready is False
    assert state.cuda.available is False
