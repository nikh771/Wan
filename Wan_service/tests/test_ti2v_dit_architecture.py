"""Tests for scripts/smoke_ti2v_dit_architecture.py (config only, no WanModel build)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ARCH_SCRIPT = REPO / "scripts" / "smoke_ti2v_dit_architecture.py"

TI2V_CONFIG = {
    "_class_name": "WanModel",
    "_diffusers_version": "0.33.0",
    "dim": 3072,
    "eps": 1e-06,
    "ffn_dim": 14336,
    "freq_dim": 256,
    "in_dim": 48,
    "model_type": "ti2v",
    "num_heads": 24,
    "num_layers": 30,
    "out_dim": 48,
    "text_len": 512,
}


@pytest.fixture(scope="module")
def arch_smoke():
    spec = importlib.util.spec_from_file_location("smoke_ti2v_dit_architecture", ARCH_SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["smoke_ti2v_dit_architecture"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_init_kwargs_from_checkpoint_config(arch_smoke) -> None:
    kwargs = arch_smoke.init_kwargs_from_checkpoint_config(dict(TI2V_CONFIG))
    assert kwargs == {
        "dim": 3072,
        "eps": 1e-06,
        "ffn_dim": 14336,
        "freq_dim": 256,
        "in_dim": 48,
        "model_type": "ti2v",
        "num_heads": 24,
        "num_layers": 30,
        "out_dim": 48,
        "text_len": 512,
    }
    # Defaults applied by from_config, not stored in checkpoint JSON:
    assert "patch_size" not in kwargs
    assert "text_dim" not in kwargs


def test_inspect_config_reads_json(tmp_path: Path, arch_smoke) -> None:
    ckpt = tmp_path / "Wan2.2-TI2V-5B"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(json.dumps(TI2V_CONFIG), encoding="utf-8")
    info = arch_smoke.inspect_config(ckpt)
    assert info.raw["dim"] == 3072
    assert info.init_kwargs_from_json["model_type"] == "ti2v"
    assert info.index_total_bytes is None


def test_rejects_non_wan_class(arch_smoke) -> None:
    bad = dict(TI2V_CONFIG)
    bad["_class_name"] = "Other"
    with pytest.raises(ValueError, match="WanModel"):
        arch_smoke.init_kwargs_from_checkpoint_config(bad)


def test_memory_refusal_when_estimate_exceeds_ram(arch_smoke) -> None:
    reason = arch_smoke.memory_refusal_reason(20_000_000_000)
    if reason is None:
        pytest.skip("system reports enough RAM for this synthetic estimate")
    assert "reserve" in reason.lower() or "RAM" in reason


def test_real_checkpoint_config_if_present(arch_smoke) -> None:
    ckpt = REPO / "models" / "Wan2.2-TI2V-5B" / "config.json"
    if not ckpt.is_file():
        pytest.skip("checkpoint not present")
    raw = arch_smoke.read_checkpoint_config_json(ckpt)
    kwargs = arch_smoke.init_kwargs_from_checkpoint_config(raw)
    assert kwargs["num_layers"] == 30
    assert kwargs["model_type"] == "ti2v"
