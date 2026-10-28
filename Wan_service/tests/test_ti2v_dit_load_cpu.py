"""Tests for scripts/smoke_ti2v_dit_load_cpu.py (preflight only, no weight load)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CPU_SCRIPT = REPO / "scripts" / "smoke_ti2v_dit_load_cpu.py"


@pytest.fixture(scope="module")
def cpu_smoke():
    spec = importlib.util.spec_from_file_location("smoke_ti2v_dit_load_cpu", CPU_SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["smoke_ti2v_dit_load_cpu"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_ti2v_ckpt(
    ckpt: Path,
    *,
    total_size: int,
    shards: dict[str, int],
    model_type: str = "ti2v",
) -> None:
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "WanModel",
                "_diffusers_version": "0.33.0",
                "model_type": model_type,
                "dim": 3072,
                "num_layers": 30,
            }
        ),
        encoding="utf-8",
    )
    weight_map = {f"t{i}": name for i, name in enumerate(shards)}
    (ckpt / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    for name, nbytes in shards.items():
        (ckpt / name).write_bytes(b"\0" * nbytes)


def test_inspect_checkpoint_ok(tmp_path: Path, cpu_smoke) -> None:
    ckpt = tmp_path / "ckpt"
    _write_ti2v_ckpt(ckpt, total_size=1000, shards={"a.safetensors": 400, "b.safetensors": 600})
    info = cpu_smoke.inspect_checkpoint(ckpt)
    assert info.num_tensors == 2
    assert info.config["model_type"] == "ti2v"
    assert info.index_total_bytes == 1000


def test_inspect_rejects_wrong_class(tmp_path: Path, cpu_smoke) -> None:
    ckpt = tmp_path / "ckpt"
    _write_ti2v_ckpt(ckpt, total_size=10, shards={"a.safetensors": 10})
    cfg = json.loads((ckpt / "config.json").read_text())
    cfg["_class_name"] = "OtherModel"
    (ckpt / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="WanModel"):
        cpu_smoke.inspect_checkpoint(ckpt)


def test_load_refusal_16gb_mac(cpu_smoke) -> None:
    est = cpu_smoke.MemoryEstimate(
        index_total_bytes=20_000_000_000,
        estimated_resident_bytes=20_000_000_000,
        estimated_peak_during_load_bytes=25_000_000_000,
        os_reserve_bytes=2_000_000_000,
    )
    reason = cpu_smoke.load_refusal_reason(est, 16 * 1024**3, 10 * 1024**3)
    assert reason is not None
    assert "resident" in reason.lower() or "ram" in reason.lower()


def test_load_refusal_none_when_ram_large(cpu_smoke) -> None:
    est = cpu_smoke.MemoryEstimate(
        index_total_bytes=5_000_000_000,
        estimated_resident_bytes=5_000_000_000,
        estimated_peak_during_load_bytes=8_000_000_000,
        os_reserve_bytes=2_000_000_000,
    )
    assert cpu_smoke.load_refusal_reason(est, 64 * 1024**3, 32 * 1024**3) is None


def test_run_check_only_refuses_tight_memory(tmp_path: Path, cpu_smoke, monkeypatch: pytest.MonkeyPatch) -> None:
    ckpt = tmp_path / "ckpt"
    _write_ti2v_ckpt(
        ckpt,
        total_size=20_000_000_000,
        shards={"p1.safetensors": 100, "p2.safetensors": 100, "p3.safetensors": 100},
    )
    monkeypatch.setattr(cpu_smoke, "system_ram_bytes", lambda: (16 * 1024**3, 8 * 1024**3))
    monkeypatch.setattr(
        cpu_smoke,
        "build_loading_plan",
        lambda: cpu_smoke.LoadingPlan(
            method="test",
            torch_dtype_name="bfloat16",
            low_cpu_mem_usage=True,
            device_map="cpu",
            accelerate_available=True,
            diffusers_version="0.39.0",
        ),
    )
    assert cpu_smoke.run_check_only(ckpt) == 2
