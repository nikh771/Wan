"""Device probes and inference readiness (CUDA vs MPS are reported separately)."""

from __future__ import annotations

import platform
import shutil
from dataclasses import dataclass
from typing import Any

from wan_service.core.config import RuntimeMode, get_settings
from wan_service.core.inference_deps import (
    probe_inference_dependencies,
    probe_wan_python,
    torch_optional_probe_cuda,
    torch_optional_probe_mps,
)

# Upstream Wan 2.2 TI2V-5B hardcodes CUDA (see wan/textimage2video.py, generate.py).
UPSTREAM_TI2V5B_MPS_SUPPORTED = False

MAC_DEV_INFERENCE_REASON = (
    "mac_dev is API/test-only; real Wan TI2V-5B inference requires linux_cuda with "
    "NVIDIA CUDA, checkpoints, and requirements-linux-cuda.txt in WAN_PYTHON."
)


@dataclass(frozen=True)
class CudaProbe:
    available: bool
    detail: str
    vram_gb: float | None
    vram_sufficient: bool | None


@dataclass(frozen=True)
class MpsProbe:
    available: bool
    detail: str


@dataclass(frozen=True)
class InferenceReadiness:
    ready: bool
    detail: str
    public_reason: str
    runtime_mode: RuntimeMode
    checkpoint_ready: bool
    generate_script_ready: bool
    ffmpeg_ready: bool
    wan_python_ready: bool
    inference_dependencies_ready: bool | None
    cuda: CudaProbe
    mps: MpsProbe


def probe_cuda() -> CudaProbe:
    available, detail, vram_gb, vram_ok = torch_optional_probe_cuda()
    return CudaProbe(
        available=available,
        detail=detail,
        vram_gb=vram_gb,
        vram_sufficient=vram_ok,
    )


def probe_mps() -> MpsProbe:
    available, detail = torch_optional_probe_mps()
    return MpsProbe(available=available, detail=detail)


def probe_ffmpeg() -> bool:
    from wan_service.utils.ffmpeg import ffmpeg_available

    return ffmpeg_available()


def checkpoint_ready() -> bool:
    settings = get_settings()
    ckpt = settings.wan_ckpt_dir
    if not ckpt.is_dir():
        return False
    try:
        return any(ckpt.iterdir())
    except OSError:
        return False


def generate_script_ready() -> bool:
    return get_settings().wan_generate_script.is_file()


def evaluate_inference_readiness() -> InferenceReadiness:
    settings = get_settings()
    mode = settings.resolved_runtime_mode()
    cuda = probe_cuda()
    mps = probe_mps()
    ckpt_ok = checkpoint_ready()
    script_ok = generate_script_ready()
    ffmpeg_ok = probe_ffmpeg()
    min_vram = settings.WAN_MIN_VRAM_GB
    py_ok, _py_detail = probe_wan_python()
    deps_ok: bool | None = None

    if mode == "mac_dev":
        detail = MAC_DEV_INFERENCE_REASON
        if mps.available and not cuda.available:
            detail += " MPS may be available but upstream TI2V-5B is CUDA-only."
        public = MAC_DEV_INFERENCE_REASON
        ready = False
        deps_ok = None
    elif platform.system() != "Linux":
        detail = f"linux_cuda mode requires Linux; this host is {platform.system()}."
        public = "linux_cuda mode requires a Linux host with NVIDIA CUDA."
        ready = False
        deps_ok = False
    elif not script_ok:
        detail = f"generate.py not found at {settings.wan_generate_script}"
        public = "Wan generate.py is missing or not configured."
        ready = False
        deps_ok = False
    elif not py_ok:
        detail = f"WAN_PYTHON is not configured correctly: {settings.WAN_PYTHON!r}"
        public = "Configured WAN_PYTHON interpreter is missing or not executable."
        ready = False
        deps_ok = False
    else:
        deps_ready, deps_detail = probe_inference_dependencies()
        deps_ok = deps_ready
        if not deps_ready:
            detail = deps_detail
            public = deps_detail
            ready = False
        elif not ckpt_ok:
            detail = f"Checkpoint directory missing or empty: {settings.wan_ckpt_dir}"
            public = "Model checkpoint directory is missing or empty."
            ready = False
        elif not cuda.available:
            detail = cuda.detail
            public = "CUDA is required for Wan 2.2 TI2V-5B on linux_cuda hosts."
            ready = False
        elif cuda.vram_sufficient is False:
            detail = (
                f"GPU VRAM below configured minimum ({min_vram:.1f} GB heuristic); "
                "generation may still OOM."
            )
            public = (
                f"GPU VRAM may be below the configured minimum ({min_vram:.1f} GB); "
                "this check does not guarantee success."
            )
            ready = False
        else:
            detail = "Wan 2.2 TI2V-5B inference stack is ready."
            public = detail
            ready = True

    return InferenceReadiness(
        ready=ready,
        detail=detail,
        public_reason=public,
        runtime_mode=mode,
        checkpoint_ready=ckpt_ok,
        generate_script_ready=script_ok,
        ffmpeg_ready=ffmpeg_ok,
        wan_python_ready=py_ok,
        inference_dependencies_ready=deps_ok,
        cuda=cuda,
        mps=mps,
    )


def _persistent_worker_health() -> dict[str, Any]:
    settings = get_settings()
    if not settings.WAN_PERSISTENT_INFERENCE:
        return {
            "persistent_inference_enabled": False,
            "persistent_worker_ready": None,
            "persistent_worker_error": None,
        }
    try:
        from wan_service.apps.jobs.services.persistent_worker import (
            get_persistent_worker,
        )

        worker = get_persistent_worker()
        return {
            "persistent_inference_enabled": True,
            "persistent_worker_ready": worker.is_ready,
            "persistent_worker_error": worker.init_error,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "persistent_inference_enabled": True,
            "persistent_worker_ready": False,
            "persistent_worker_error": str(exc),
        }


def wan_health_payload() -> dict[str, Any]:
    """Wan TI2V-5B readiness (same shape as ltx/cogvideox health)."""
    settings = get_settings()
    state = evaluate_inference_readiness()
    return {
        "engine": "wan",
        "model": "Wan2.2-TI2V-5B",
        "ready": state.ready,
        "status": "ready" if state.ready else "unavailable",
        "detail": None if state.ready else state.public_reason,
        "runtime_mode": state.runtime_mode,
        "inference_backend": settings.resolved_inference_backend(),
        "checkpoint_ready": state.checkpoint_ready,
        "generate_script_ready": state.generate_script_ready,
        "wan_python_ready": state.wan_python_ready,
        "inference_dependencies_ready": state.inference_dependencies_ready,
        "cuda_available": state.cuda.available,
        "cuda_detail": state.cuda.detail,
        "vram_gb": state.cuda.vram_gb,
        "vram_min_gb": settings.WAN_MIN_VRAM_GB,
        "vram_sufficient": state.cuda.vram_sufficient,
        "mps_available": state.mps.available,
        "upstream_ti2v5b_mps_supported": UPSTREAM_TI2V5B_MPS_SUPPORTED,
        "ffmpeg_ready": state.ffmpeg_ready,
        **_persistent_worker_health(),
    }


def engine_health_payload(engine_type: str) -> dict[str, Any]:
    """Health for one engine (``wan`` | ``ltx`` | ``cogvideox``)."""
    from core.engine_dispatch import normalize_engine

    from cogvideox_service.health import cogvideox_health_payload
    from ltx_service.health import ltx_health_payload

    name = normalize_engine(engine_type)
    if name == "wan":
        body = wan_health_payload()
    elif name == "ltx":
        body = ltx_health_payload()
    else:
        body = cogvideox_health_payload()
    ready = bool(body.get("ready"))
    out: dict[str, Any] = {
        "status": "ok" if ready else "degraded",
        "engine_type": name,
        "service_ready": True,
        **body,
    }
    if name == "wan":
        out["inference_ready"] = ready
        out["inference_unavailable_reason"] = body.get("detail")
    return out


def assert_inference_available() -> None:
    state = evaluate_inference_readiness()
    if not state.ready:
        raise RuntimeError(state.detail)
