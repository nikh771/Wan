"""Probe Wan inference interpreter and packages (linux_cuda only; never imports flash_attn here)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from wan_service.core.config import get_settings

# Modules required for upstream TI2V-5B generate.py (checked in WAN_PYTHON subprocess).
WAN_INFERENCE_MODULES_CUDA: tuple[str, ...] = (
    "torch",
    "transformers",
    "diffusers",
    "easydict",
    "flash_attn",
)

WAN_INFERENCE_MODULES_MPS: tuple[str, ...] = (
    "torch",
    "transformers",
    "diffusers",
    "easydict",
)


def _inference_modules_for_backend() -> tuple[str, ...]:
    import os

    backend = os.environ.get("WAN_INFERENCE_BACKEND", "").strip().lower()
    if backend == "mac_mps":
        return WAN_INFERENCE_MODULES_MPS
    return WAN_INFERENCE_MODULES_CUDA


def resolve_wan_python() -> Path | None:
    settings = get_settings()
    raw = settings.WAN_PYTHON.strip() or "python"
    found = shutil.which(raw)
    if found:
        return Path(found).resolve()
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = settings.project_root / candidate
    if candidate.is_file():
        return candidate.resolve()
    return None


def probe_wan_python() -> tuple[bool, str]:
    interpreter = resolve_wan_python()
    if interpreter is None:
        settings = get_settings()
        return False, f"WAN_PYTHON not found or not executable: {settings.WAN_PYTHON!r}"
    try:
        result = subprocess.run(
            [str(interpreter), "-c", "import sys; print(sys.executable)"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except OSError as exc:
        return False, f"WAN_PYTHON failed to start: {exc}"
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip()[-300:]
        return False, f"WAN_PYTHON is not usable: {tail or 'unknown error'}"
    return True, result.stdout.strip() or str(interpreter)


def probe_inference_dependencies() -> tuple[bool, str]:
    """Verify packages in the configured inference interpreter (subprocess only)."""
    interpreter = resolve_wan_python()
    if interpreter is None:
        settings = get_settings()
        return False, f"WAN_PYTHON not found: {settings.WAN_PYTHON!r}"

    settings = get_settings()
    repo = settings.wan_repo_dir
    modules = _inference_modules_for_backend()
    imports = "; ".join(
        f"importlib.import_module({name!r})" for name in modules
    )
    code = f"import importlib; {imports}"
    env = os.environ.copy()
    prefix = str(repo)
    env["PYTHONPATH"] = prefix + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    try:
        result = subprocess.run(
            [str(interpreter), "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(repo) if repo.is_dir() else None,
            env=env,
            check=False,
        )
    except OSError as exc:
        return False, f"Dependency check failed to run: {exc}"

    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        if "flash_attn" in err or "ModuleNotFoundError" in err:
            return (
                False,
                "Wan inference packages missing in WAN_PYTHON (install requirements-linux-cuda.txt).",
            )
        return False, "Wan inference dependency check failed for WAN_PYTHON."

    return True, "Wan inference dependencies available in WAN_PYTHON."


def torch_optional_probe_cuda() -> tuple[bool, str, float | None, bool | None]:
    """CUDA probe without requiring torch in the API process."""
    try:
        import torch
    except ImportError:
        return (
            False,
            "PyTorch not installed in API process (install CUDA torch on linux_cuda hosts)",
            None,
            None,
        )

    settings = get_settings()
    min_gb = settings.WAN_MIN_VRAM_GB
    if not torch.cuda.is_available():
        return False, "CUDA not available", None, False

    name = torch.cuda.get_device_name(0)
    total_gb: float | None = None
    try:
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024**3)
    except Exception:  # noqa: BLE001
        total_gb = None

    vram_ok = None if total_gb is None else total_gb >= min_gb
    detail = name if total_gb is None else f"{name} ({total_gb:.1f} GB)"
    return True, detail, total_gb, vram_ok


def torch_optional_probe_mps() -> tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "PyTorch not installed (optional for mac_dev API-only mode)"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None:
        return False, "PyTorch build has no MPS backend"
    if mps_backend.is_available():
        return True, "Apple Metal Performance Shaders (MPS)"
    return False, "MPS not available on this host"
