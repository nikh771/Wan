"""CUDA probe (legacy module; see wan_service.core.runtime)."""

from __future__ import annotations

from wan_service.core.runtime import probe_cuda as _probe_cuda


def probe_cuda() -> tuple[bool, str, bool | None]:
    """Return (cuda_ok, detail, vram_sufficient_for_ti2v_5b)."""
    result = _probe_cuda()
    return result.available, result.detail, result.vram_sufficient  # vram heuristic uses WAN_MIN_VRAM_GB
