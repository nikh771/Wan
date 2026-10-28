"""Unit tests for DiT load smoke helpers (no weight load)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = REPO / "scripts" / "smoke_ti2v_dit_load_mps.py"


@pytest.fixture(scope="module")
def smoke_mod():
    spec = importlib.util.spec_from_file_location("smoke_ti2v_dit", SMOKE_SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["smoke_ti2v_dit"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_dit_shard_paths_from_index(tmp_path: Path, smoke_mod) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}", encoding="utf-8")
    index = {
        "metadata": {"total_size": 1000},
        "weight_map": {
            "blocks.0.foo": "part_a.safetensors",
            "blocks.1.foo": "part_b.safetensors",
        },
    }
    (ckpt / smoke_mod.INDEX_NAME).write_text(json.dumps(index), encoding="utf-8")
    (ckpt / "part_a.safetensors").write_bytes(b"x" * 400)
    (ckpt / "part_b.safetensors").write_bytes(b"y" * 600)

    shards, total = smoke_mod.validate_dit_checkpoint(ckpt)
    assert len(shards) == 2
    assert total >= 1000


def test_validate_missing_shard_raises(tmp_path: Path, smoke_mod) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}", encoding="utf-8")
    index = {"metadata": {"total_size": 10}, "weight_map": {"a": "missing.safetensors"}}
    (ckpt / smoke_mod.INDEX_NAME).write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="missing.safetensors"):
        smoke_mod.validate_dit_checkpoint(ckpt)


def test_memory_blocker_tight_ram(smoke_mod, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(smoke_mod, "_system_ram_bytes", lambda: (16 * 1024**3, 8 * 1024**3))
    msg = smoke_mod.memory_blocker(20 * 1024**3)
    assert msg is not None
    assert "BLOCKER" in msg or "not safe" in msg or "exceeds" in msg


def test_memory_blocker_ok(smoke_mod, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        smoke_mod,
        "_system_ram_bytes",
        lambda: (64 * 1024**3, 32 * 1024**3),
    )
    assert smoke_mod.memory_blocker(5 * 1024**3) is None


def test_memory_blocker_large_ckpt_without_ram_info(smoke_mod, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(smoke_mod, "_system_ram_bytes", lambda: (None, None))
    msg = smoke_mod.memory_blocker(20 * 1024**3)
    assert msg is not None
