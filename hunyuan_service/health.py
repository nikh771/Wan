"""Hunyuan health without loading weights (API process safe)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from hunyuan_service.config import (
    HUNYUAN_HF_ID,
    HUNYUAN_MODEL_ID,
    model_available,
    repo_ready,
)


def _hunyuan_python() -> Path | None:
    from hunyuan_service.settings import get_hunyuan_settings

    cfg = get_hunyuan_settings()
    raw = cfg.resolved_hunyuan_python()
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = cfg.project_root / candidate
    if candidate.is_file():
        return candidate
    found = shutil.which(raw)
    return Path(found) if found else None


def probe_hunyuan_cuda() -> dict[str, Any]:
    py = _hunyuan_python()
    if py is None:
        return {"cuda_available": False, "gpu": None, "error": "HUNYUAN_PYTHON not found"}
    code = (
        "import json,torch; "
        "d={'cuda_available':torch.cuda.is_available(),'pytorch':torch.__version__}; "
        "d['gpu']=torch.cuda.get_device_name(0) if d['cuda_available'] else None; "
        "print(json.dumps(d))"
    )
    try:
        out = subprocess.run(
            [str(py), "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if out.returncode != 0:
            return {"cuda_available": False, "gpu": None, "error": (out.stderr or "")[-300:]}
        return json.loads((out.stdout or "{}").strip() or "{}")
    except (OSError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        return {"cuda_available": False, "gpu": None, "error": str(exc)}


def load_capability_registry() -> dict[str, Any]:
    path = Path(__file__).resolve().parent / "capability_registry.yaml"
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def hunyuan_health_payload() -> dict[str, Any]:
    from hunyuan_service.settings import get_hunyuan_settings

    cfg = get_hunyuan_settings()
    repo = cfg.hunyuan_repo_dir
    model = cfg.hunyuan_model_path
    repo_ok = repo_ready(repo)
    weights_ok = model_available(model)
    cuda = probe_hunyuan_cuda()
    py_ok = _hunyuan_python() is not None
    ready = repo_ok and weights_ok and py_ok and bool(cuda.get("cuda_available"))
    caps = load_capability_registry().get("hunyuan_1_5", {})
    return {
        "engine": "hunyuan",
        "ready": ready,
        "status": "ready" if ready else "unavailable",
        "model": HUNYUAN_MODEL_ID,
        "hf_id": HUNYUAN_HF_ID,
        "repo_ready": repo_ok,
        "model_available": weights_ok,
        "repo_dir": str(repo),
        "model_path": str(model),
        "hunyuan_python": cfg.resolved_hunyuan_python(),
        "hunyuan_python_ok": py_ok,
        "cuda_available": bool(cuda.get("cuda_available")),
        "gpu": cuda.get("gpu"),
        "pytorch": cuda.get("pytorch"),
        "resolution": cfg.resolved_resolution(),
        "prompt_rewrite": bool(cfg.HUNYUAN_PROMPT_REWRITE),
        "enable_offload": bool(cfg.HUNYUAN_ENABLE_OFFLOAD),
        "capabilities": caps.get("capabilities", ["text_to_video", "image_to_video"]),
        "detail": None if ready else _unavailable_reason(repo_ok, weights_ok, py_ok, cuda),
    }


def _unavailable_reason(repo_ok: bool, model_ok: bool, py_ok: bool, cuda: dict) -> str:
    parts: list[str] = []
    if not repo_ok:
        parts.append("HunyuanVideo-1.5 source missing (clone to HUNYUAN_REPO_DIR)")
    if not model_ok:
        parts.append(f"checkpoints missing at HUNYUAN_MODEL_PATH (hf download {HUNYUAN_HF_ID})")
    if not py_ok:
        parts.append("HUNYUAN_PYTHON not configured")
    if not cuda.get("cuda_available"):
        parts.append("CUDA not available in HUNYUAN_PYTHON")
    return "; ".join(parts) or "unknown"