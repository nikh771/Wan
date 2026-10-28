# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Wan_service: device backend for TI2V-5B (mac_mps | linux_cuda).
"""Resolve inference device, autocast, sync, and dtype without CUDA-only assumptions."""

from __future__ import annotations

import gc
import os
from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn

# Accepted anywhere Wan passes a load/compute device (str or torch.device).
DeviceLike = str | torch.device

VALID_BACKENDS = frozenset({"mac_mps", "linux_cuda"})

__all__ = [
    "DeviceLike",
    "VALID_BACKENDS",
    "get_inference_backend",
    "resolve_inference_device",
    "autocast_device_type",
    "param_dtype_for_backend",
    "mps_supports_bfloat16",
    "resolve_autocast_dtype",
    "rope_float_dtype",
    "inference_autocast",
    "sync_device",
    "empty_device_cache",
    "is_mac_mps_backend",
    "assert_ti2v_single_process_compatible",
    "init_distributed_if_cuda",
    "create_inference_generator",
    "offload_module_to_cpu",
    "should_offload_between_steps",
]


def get_inference_backend() -> str:
    raw = os.environ.get("WAN_INFERENCE_BACKEND", "").strip().lower()
    if raw in VALID_BACKENDS:
        return raw
    if raw:
        raise RuntimeError(
            f"Unsupported WAN_INFERENCE_BACKEND={raw!r}. Use mac_mps or linux_cuda."
        )
    import platform

    return "mac_mps" if platform.system() == "Darwin" else "linux_cuda"


def resolve_inference_device(device_id: int = 0) -> torch.device:
    backend = get_inference_backend()
    if backend == "mac_mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("WAN_INFERENCE_BACKEND=mac_mps but MPS is not available.")
        return torch.device("mps")
    if not torch.cuda.is_available():
        raise RuntimeError("WAN_INFERENCE_BACKEND=linux_cuda but CUDA is not available.")
    return torch.device(f"cuda:{device_id}")


def autocast_device_type(device: torch.device | None = None) -> str:
    if device is not None:
        return device.type
    backend = get_inference_backend()
    return "mps" if backend == "mac_mps" else "cuda"


def mps_supports_bfloat16() -> bool:
    is_bf16 = getattr(torch.backends.mps, "is_bfloat16_supported", None)
    return bool(callable(is_bf16) and is_bf16())


def param_dtype_for_backend(requested: torch.dtype) -> torch.dtype:
    """Pick a safe compute dtype for DiT/VAE on the active backend."""
    if get_inference_backend() != "mac_mps":
        return requested
    if requested == torch.bfloat16:
        if mps_supports_bfloat16():
            return torch.bfloat16
        return torch.float16
    return requested


def resolve_autocast_dtype(device: torch.device, requested: torch.dtype) -> torch.dtype:
    """Map config dtype to a dtype safe for ``torch.amp.autocast`` on *device*."""
    if device.type == "cpu":
        return requested
    if device.type == "mps":
        return param_dtype_for_backend(requested)
    return requested


def is_mac_mps_backend() -> bool:
    return get_inference_backend() == "mac_mps"


def assert_ti2v_single_process_compatible(
    *,
    t5_fsdp: bool,
    dit_fsdp: bool,
    use_sp: bool,
) -> None:
    """Block CUDA-only distributed options on the Mac MPS TI2V path."""
    if not is_mac_mps_backend():
        return
    if t5_fsdp or dit_fsdp or use_sp:
        raise RuntimeError(
            "TI2V-5B on mac_mps does not support t5_fsdp, dit_fsdp, or sequence "
            "parallelism (ulysses). Use single-process inference with "
            "WAN_INFERENCE_BACKEND=mac_mps."
        )


def rope_float_dtype(device: torch.device) -> torch.dtype:
    if device.type == "mps":
        return torch.float32
    return torch.float64


@contextmanager
def inference_autocast(
    device: torch.device,
    dtype: torch.dtype,
    *,
    enabled: bool = True,
) -> Iterator[None]:
    if not enabled or device.type == "cpu":
        yield
        return
    cast_dtype = resolve_autocast_dtype(device, dtype)
    device_type = autocast_device_type(device)
    with torch.amp.autocast(
        device_type=device_type, dtype=cast_dtype, enabled=True
    ):
        yield


def init_distributed_if_cuda(
    world_size: int,
    local_rank: int,
    rank: int,
) -> None:
    """Initialize NCCL only on the Linux CUDA backend (multi-GPU)."""
    if world_size <= 1:
        return
    if get_inference_backend() != "linux_cuda":
        raise RuntimeError(
            "Distributed inference (WORLD_SIZE>1) requires "
            "WAN_INFERENCE_BACKEND=linux_cuda and NVIDIA CUDA."
        )
    import torch.distributed as dist

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )


def create_inference_generator(device: torch.device) -> torch.Generator:
    """RNG for scheduler / noise ops on the inference accelerator when supported."""
    if device.type == "cuda":
        return torch.Generator(device=device)
    if device.type == "mps":
        try:
            return torch.Generator(device=torch.device("mps"))
        except RuntimeError:
            return torch.Generator(device=torch.device("cpu"))
    return torch.Generator(device=torch.device("cpu"))


def offload_module_to_cpu(module: nn.Module) -> None:
    """Backend-neutral VRAM offload (CUDA / MPS / CPU)."""
    module.cpu()


def should_offload_between_steps(
    offload_model: bool,
    accelerator: torch.device,
) -> bool:
    """Whether to ping-pong weights to CPU between denoise steps."""
    if not offload_model:
        return False
    return accelerator.type in ("cuda", "mps")


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def empty_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        gc.collect()
        torch.mps.synchronize()
