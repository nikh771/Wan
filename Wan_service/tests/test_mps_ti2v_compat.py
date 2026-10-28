"""TI2V-5B MPS compatibility (no weights, no full inference)."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

REPO = Path(__file__).resolve().parents[1]
WAN_ROOT = REPO / "third_party" / "Wan2.2"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def device_compat():
    return _load_module(
        "wan_device_compat_test",
        WAN_ROOT / "wan" / "device_compat.py",
    )


def test_inference_backend_mac_mps(device_compat) -> None:
    os.environ["WAN_INFERENCE_BACKEND"] = "mac_mps"
    assert device_compat.get_inference_backend() == "mac_mps"
    pytest.importorskip("torch")
    import torch

    if torch.backends.mps.is_available():
        assert device_compat.resolve_inference_device().type == "mps"


def test_inference_backend_linux_cuda(device_compat, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WAN_INFERENCE_BACKEND", "linux_cuda")
    assert device_compat.get_inference_backend() == "linux_cuda"


def test_param_dtype_mps_fallback(device_compat, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    import torch

    monkeypatch.setenv("WAN_INFERENCE_BACKEND", "mac_mps")
    out = device_compat.param_dtype_for_backend(torch.bfloat16)
    if device_compat.mps_supports_bfloat16():
        assert out == torch.bfloat16
    else:
        assert out == torch.float16


def test_resolve_autocast_dtype_cpu_passthrough(device_compat) -> None:
    pytest.importorskip("torch")
    import torch

    dev = torch.device("cpu")
    assert device_compat.resolve_autocast_dtype(dev, torch.bfloat16) == torch.bfloat16


def test_sync_and_cache_helpers_no_crash(device_compat) -> None:
    pytest.importorskip("torch")
    import torch

    for dev_name in ("cpu", "mps", "cuda"):
        if dev_name == "mps" and not torch.backends.mps.is_available():
            continue
        if dev_name == "cuda" and not torch.cuda.is_available():
            continue
        dev = torch.device(dev_name)
        device_compat.sync_device(dev)
        device_compat.empty_device_cache(dev)


def test_inference_autocast_context(device_compat) -> None:
    pytest.importorskip("torch")
    import torch

    dev = torch.device("cpu")
    with device_compat.inference_autocast(dev, torch.float16):
        x = torch.ones(2, dtype=torch.float32)
    assert x.shape == (2,)

    if torch.backends.mps.is_available():
        dev = torch.device("mps")
        with device_compat.inference_autocast(dev, torch.bfloat16):
            y = torch.ones(2, device=dev, dtype=torch.float32)
        assert y.device.type == "mps"


def test_create_inference_generator(device_compat) -> None:
    pytest.importorskip("torch")
    import torch

    gen = device_compat.create_inference_generator(torch.device("cpu"))
    gen.manual_seed(0)
    a = torch.randn(4, generator=gen)
    gen.manual_seed(0)
    b = torch.randn(4, generator=gen)
    assert torch.equal(a, b)

    if torch.backends.mps.is_available():
        gen_mps = device_compat.create_inference_generator(torch.device("mps"))
        gen_mps.manual_seed(1)
        n = torch.randn(3, device="mps", generator=gen_mps)
        assert n.device.type == "mps"


def test_offload_helpers(device_compat) -> None:
    pytest.importorskip("torch")
    import torch

    assert not device_compat.should_offload_between_steps(True, torch.device("cpu"))
    assert not device_compat.should_offload_between_steps(False, torch.device("cpu"))

    if torch.backends.mps.is_available():
        mps = torch.device("mps")
        assert device_compat.should_offload_between_steps(True, mps)
        assert not device_compat.should_offload_between_steps(False, mps)
        linear = torch.nn.Linear(2, 2).to(mps)
    elif torch.cuda.is_available():
        cuda = torch.device("cuda")
        assert device_compat.should_offload_between_steps(True, cuda)
        linear = torch.nn.Linear(2, 2).to(cuda)
    else:
        linear = torch.nn.Linear(2, 2)

    device_compat.offload_module_to_cpu(linear)
    assert next(linear.parameters()).device.type == "cpu"


def test_assert_ti2v_rejects_distributed_on_mac_mps(device_compat, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WAN_INFERENCE_BACKEND", "mac_mps")
    with pytest.raises(RuntimeError, match="mac_mps"):
        device_compat.assert_ti2v_single_process_compatible(
            t5_fsdp=True,
            dit_fsdp=False,
            use_sp=False,
        )
    device_compat.assert_ti2v_single_process_compatible(
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
    )


def test_flash_attention_uses_sdpa_on_mps() -> None:
    pytest.importorskip("torch")
    import torch

    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    attn_mod = _load_module(
        "wan_attention_test",
        WAN_ROOT / "wan" / "modules" / "attention.py",
    )
    q = k = v = torch.randn(1, 16, 8, 64, device="mps", dtype=torch.float16)
    out = attn_mod.flash_attention(q, k, v, dtype=torch.float16)
    assert out.shape == q.shape


def test_invalid_inference_backend_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from wan_service.core.config import Settings, clear_settings_cache

    monkeypatch.setenv("WAN_INFERENCE_BACKEND", "cuda_only")
    clear_settings_cache()
    with pytest.raises(ValidationError, match="WAN_INFERENCE_BACKEND"):
        Settings()
    monkeypatch.delenv("WAN_INFERENCE_BACKEND", raising=False)
    clear_settings_cache()
