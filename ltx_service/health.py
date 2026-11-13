"""LTX health without loading weights (API process safe)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from ltx_service.config import (
    DEFAULT_PIPELINE_REL,
    LTX_CHECKPOINT_FILENAME,
    LTX_MODEL_ID,
    default_model_path,
)


def _ltx_python() -> Path | None:
    from ltx_service.settings import get_ltx_settings

    cfg = get_ltx_settings()
    raw = cfg.resolved_ltx_python()
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = cfg.project_root / candidate
    if candidate.is_file():
        return candidate
    found = shutil.which(raw)
    return Path(found) if found else None


def probe_ltx_cuda() -> dict[str, Any]:
    py = _ltx_python()
    if py is None:
        return {"cuda_available": False, "gpu": None, "error": "LTX_PYTHON not found"}
    code = (
        "import json,torch; "
        "d={'cuda_available':torch.cuda.is_available()}; "
        "d['gpu']=torch.cuda.get_device_name(0) if d['cuda_available'] else None; "
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


def ltx_health_payload() -> dict[str, Any]:
    from ltx_service.settings import get_ltx_settings

    cfg = get_ltx_settings()
    repo = cfg.ltx_repo_dir
    model = cfg.ltx_model_path
    installed = (repo / "inference.py").is_file()
    model_available = model.is_file()
    cuda = probe_ltx_cuda()
    py_ok = _ltx_python() is not None
    ready = installed and model_available and py_ok and cuda.get("cuda_available")
    return {
        "engine": "ltx",
        "model": LTX_MODEL_ID,
        "checkpoint": LTX_CHECKPOINT_FILENAME,
        "installed": installed,
        "model_available": model_available,
        "model_path": str(model),
        "repo_dir": str(repo),
        "pipeline_config": cfg.resolved_ltx_pipeline_config(),
        "ltx_python": cfg.resolved_ltx_python(),
        "ltx_python_ok": py_ok,
        "cuda_available": bool(cuda.get("cuda_available")),
        "gpu": cuda.get("gpu"),
        "capabilities": load_capability_registry().get("ltx_2b_0_9_6_distilled", {}).get(
            "capabilities", ["text_to_video", "image_to_video"]
        ),
        "status": "ready" if ready else "unavailable",
        "detail": None if ready else _unavailable_reason(installed, model_available, py_ok, cuda),
    }


def _unavailable_reason(installed: bool, model_ok: bool, py_ok: bool, cuda: dict) -> str:
    parts: list[str] = []
    if not installed:
        parts.append("LTX-Video repo not installed")
    if not model_ok:
        parts.append(f"checkpoint missing ({LTX_CHECKPOINT_FILENAME})")
    if not py_ok:
        parts.append("LTX_PYTHON not configured")
    if not cuda.get("cuda_available"):
        parts.append("CUDA not available in LTX_PYTHON")
    return "; ".join(parts) or "unknown"


def assert_ltx_model_layout(project_root: Path | None = None) -> Path:
    from ltx_service.settings import get_ltx_settings

    return get_ltx_settings().ltx_model_path
