"""CogVideoX health without loading weights (API process safe)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from cogvideox_service.config import (
    COGVIDEOX_HF_ID,
    COGVIDEOX_MODEL_ID,
    LICENSE_METADATA,
    model_available,
)
from cogvideox_service.gpu_state import read_gpu_state


def _cogvideox_python() -> Path | None:
    from cogvideox_service.settings import get_cogvideox_settings

    cfg = get_cogvideox_settings()
    raw = cfg.resolved_cogvideox_python()
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = cfg.project_root / candidate
    if candidate.is_file():
        return candidate
    found = shutil.which(raw)
    return Path(found) if found else None


def probe_cogvideox_cuda() -> dict[str, Any]:
    py = _cogvideox_python()
    if py is None:
        return {"cuda_available": False, "gpu": None, "error": "COGVIDEOX_PYTHON not found"}
    code = (
        "import json,torch; "
        "d={'cuda_available':torch.cuda.is_available(),'pytorch':torch.__version__,"
        "'cuda_version':torch.version.cuda}; "
        "d['gpu']=torch.cuda.get_device_name(0) if d['cuda_available'] else None; "
        "d['vram_total_gb']=round(torch.cuda.get_device_properties(0).total_memory/(1024**3),2) "
        "if d['cuda_available'] else None; "
        "print(json.dumps(d))"
    )
    try:
        out = subprocess.run(
            [str(py), "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if out.returncode != 0:
            return {"cuda_available": False, "gpu": None, "error": (out.stderr or "")[-200:]}
        return json.loads((out.stdout or "{}").strip() or "{}")
    except (OSError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        return {"cuda_available": False, "gpu": None, "error": str(exc)}


def load_capability_registry() -> dict[str, Any]:
    path = Path(__file__).resolve().parent / "capability_registry.yaml"
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def cogvideox_health_payload() -> dict[str, Any]:
    from cogvideox_service.settings import get_cogvideox_settings

    cfg = get_cogvideox_settings()
    model = cfg.cogvideox_model_path
    available = model_available(model)
    cuda = probe_cogvideox_cuda()
    py_ok = _cogvideox_python() is not None
    ready = available and py_ok and bool(cuda.get("cuda_available"))
    caps = load_capability_registry().get("cogvideox_2b", {})
    gpu_state = read_gpu_state()
    loaded = gpu_state.get("state") in ("COGVIDEOX_LOADED", "GENERATING") and (
        gpu_state.get("owner") == "cogvideox"
    )
    return {
        "engine": "cogvideox",
        "ready": ready,
        "status": "ready" if ready else "unavailable",
        "model": COGVIDEOX_MODEL_ID,
        "hf_id": cfg.COGVIDEOX_HF_ID or COGVIDEOX_HF_ID,
        "loaded": bool(loaded),
        "cuda_available": bool(cuda.get("cuda_available")),
        "gpu": cuda.get("gpu"),
        "vram_total_gb": cuda.get("vram_total_gb"),
        "pytorch": cuda.get("pytorch"),
        "cuda_version": cuda.get("cuda_version"),
        "model_available": available,
        "model_path": str(model),
        "cogvideox_python": cfg.resolved_cogvideox_python(),
        "cogvideox_python_ok": py_ok,
        "memory_mode": cfg.resolved_memory_mode(),
        "gpu_state": gpu_state.get("state"),
        "capabilities": caps.get("modes", ["text_to_video"]),
        "license": LICENSE_METADATA,
        "detail": None if ready else _unavailable_reason(available, py_ok, cuda),
    }


def _unavailable_reason(model_ok: bool, py_ok: bool, cuda: dict) -> str:
    parts: list[str] = []
    if not model_ok:
        parts.append(
            f"model missing at COGVIDEOX_MODEL_PATH (download {COGVIDEOX_HF_ID})"
        )
    if not py_ok:
        parts.append("COGVIDEOX_PYTHON not configured")
    if not cuda.get("cuda_available"):
        parts.append("CUDA not available in COGVIDEOX_PYTHON")
    return "; ".join(parts) or "unknown"
