#!/usr/bin/env python3
"""
CPU-only DiT weight load diagnostic for Wan 2.2 TI2V-5B (no T5/VAE/inference, no MPS/CUDA).

Uses upstream ``wan.modules.model.WanModel.from_pretrained`` (Diffusers ``ModelMixin``).

  cd Wan_service
  .venv/bin/python scripts/smoke_ti2v_dit_load_cpu.py --check-only
  .venv/bin/python scripts/smoke_ti2v_dit_load_cpu.py --ckpt-dir models/Wan2.2-TI2V-5B

Success (full run): exit 0 after ``from_pretrained`` completes with all parameters on CPU.
Refusal: exit 2 when preflight decides load is unsafe (no retry).
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import importlib.util
import json
import logging
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
WAN_ROOT = REPO_ROOT / "third_party" / "Wan2.2"
DEFAULT_CKPT = REPO_ROOT / "models" / "Wan2.2-TI2V-5B"
INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"
# Reserve for macOS + Python runtime (not a substitute for fitting the full model).
OS_RESERVE_BYTES = 2 * 1024**3
# Extra headroom during load (Diffusers may spike; we do not assume shard streaming removes final footprint).
LOAD_SPIKE_RATIO = 1.15


@dataclass(frozen=True)
class CheckpointInfo:
    ckpt_dir: Path
    config: dict[str, Any]
    num_tensors: int
    shards: tuple[Path, ...]
    index_total_bytes: int
    shard_bytes_on_disk: int


@dataclass(frozen=True)
class MemoryEstimate:
    index_total_bytes: int
    estimated_resident_bytes: int
    estimated_peak_during_load_bytes: int
    os_reserve_bytes: int


@dataclass(frozen=True)
class LoadingPlan:
    method: str
    torch_dtype_name: str
    low_cpu_mem_usage: bool
    device_map: str
    accelerate_available: bool
    diffusers_version: str


def _rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def system_ram_bytes() -> tuple[int | None, int | None]:
    try:
        import psutil  # type: ignore[import-not-found]

        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available)
    except ImportError:
        pass
    try:
        total = int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        )
        return total, None
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None, None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_wan_config(config: dict[str, Any]) -> None:
    if config.get("_class_name") != "WanModel":
        raise ValueError(
            f"config.json _class_name={config.get('_class_name')!r}, expected 'WanModel'"
        )
    if config.get("model_type") != "ti2v":
        raise ValueError(
            f"config.json model_type={config.get('model_type')!r}, expected 'ti2v'"
        )


def inspect_checkpoint(ckpt_dir: Path) -> CheckpointInfo:
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    config_path = ckpt_dir / "config.json"
    index_path = ckpt_dir / INDEX_NAME
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing {config_path}")
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing {index_path}")

    config = _read_json(config_path)
    validate_wan_config(config)

    index = _read_json(index_path)
    weight_map = index.get("weight_map") or {}
    num_tensors = len(weight_map)
    shard_names = sorted(set(weight_map.values()))
    shards = tuple(ckpt_dir / name for name in shard_names)
    missing = [p for p in shards if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing DiT shard(s): "
            + ", ".join(p.name for p in missing)
        )

    meta = index.get("metadata") or {}
    index_total = int(meta.get("total_size") or 0)
    on_disk = sum(p.stat().st_size for p in shards)
    if index_total <= 0:
        index_total = on_disk

    return CheckpointInfo(
        ckpt_dir=ckpt_dir,
        config=config,
        num_tensors=num_tensors,
        shards=shards,
        index_total_bytes=index_total,
        shard_bytes_on_disk=on_disk,
    )


def estimate_memory(info: CheckpointInfo) -> MemoryEstimate:
    """
    Conservative RAM estimate for a full CPU-resident DiT after load.

    ``low_cpu_mem_usage`` can lower *peak during* loading; it does not shrink the
    final parameter footprint on CPU, which we approximate from index ``total_size``.
    """
    resident = max(info.index_total_bytes, info.shard_bytes_on_disk)
    peak = int(resident * LOAD_SPIKE_RATIO) + OS_RESERVE_BYTES
    return MemoryEstimate(
        index_total_bytes=info.index_total_bytes,
        estimated_resident_bytes=resident,
        estimated_peak_during_load_bytes=peak,
        os_reserve_bytes=OS_RESERVE_BYTES,
    )


def load_refusal_reason(
    estimate: MemoryEstimate,
    total_ram: int | None,
    available: int | None,
) -> str | None:
    resident_need = estimate.estimated_resident_bytes + estimate.os_reserve_bytes
    peak_need = estimate.estimated_peak_during_load_bytes

    if total_ram is not None and resident_need > total_ram:
        return (
            f"Estimated CPU-resident DiT size (~{estimate.estimated_resident_bytes / 1e9:.2f} GB "
            f"from checkpoint index, plus ~{estimate.os_reserve_bytes / 1e9:.1f} GB OS reserve) "
            f"exceeds system RAM ({total_ram / 1e9:.2f} GB). "
            "Diffusers low_cpu_mem_usage can reduce load-time peaks but does not remove the "
            "need to hold the full weight tensors on CPU after load."
        )
    if total_ram is not None and peak_need > total_ram:
        return (
            f"Estimated load peak (~{peak_need / 1e9:.2f} GB, including reserve) exceeds "
            f"system RAM ({total_ram / 1e9:.2f} GB)."
        )
    if available is not None and peak_need > available:
        return (
            f"Estimated load peak (~{peak_need / 1e9:.2f} GB) exceeds currently available RAM "
            f"({available / 1e9:.2f} GB). Free memory or use --check-only."
        )
    return None


def _import_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def ti2v_param_dtype():
    """Official TI2V ``param_dtype`` from ``wan/configs/shared_config.py`` (same as ``ti2v-5B`` config)."""
    shared = _import_module_from_path(
        "wan_shared_config_smoke",
        WAN_ROOT / "wan" / "configs" / "shared_config.py",
    )
    return shared.wan_shared_cfg.param_dtype


def import_wan_model_class():
    """``WanModel`` via ``wan.modules.model`` (avoids ``wan/__init__.py`` side effects)."""
    if str(WAN_ROOT) not in sys.path:
        sys.path.insert(0, str(WAN_ROOT))
    _import_module_from_path(
        "wan_model_smoke",
        WAN_ROOT / "wan" / "modules" / "model.py",
    )
    from wan.modules.model import WanModel

    return WanModel


def build_loading_plan() -> LoadingPlan:
    try:
        diffusers_version = importlib.metadata.version("diffusers")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("diffusers is not installed") from exc

    from diffusers.utils import is_accelerate_available

    accelerate_ok = is_accelerate_available()
    if not accelerate_ok:
        raise RuntimeError(
            "Accelerate is required for low_cpu_mem_usage loading with this Diffusers version. "
            "Install accelerate (already listed in requirements-mps-inference.txt)."
        )

    torch_dtype = ti2v_param_dtype()
    dtype_name = str(torch_dtype).replace("torch.", "")

    method = (
        "WanModel.from_pretrained(checkpoint_dir, "
        f"torch_dtype=torch.{dtype_name}, "
        "low_cpu_mem_usage=True, device_map='cpu')"
    )
    return LoadingPlan(
        method=method,
        torch_dtype_name=dtype_name,
        low_cpu_mem_usage=True,
        device_map="cpu",
        accelerate_available=accelerate_ok,
        diffusers_version=diffusers_version,
    )


def load_dit_cpu(info: CheckpointInfo, plan: LoadingPlan):
    WanModel = import_wan_model_class()
    torch_dtype = ti2v_param_dtype()
    if not plan.low_cpu_mem_usage or not plan.accelerate_available:
        raise RuntimeError(
            "Refusing high-memory load: low_cpu_mem_usage requires Accelerate "
            f"(accelerate_available={plan.accelerate_available})."
        )

    model = WanModel.from_pretrained(
        str(info.ckpt_dir),
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    model.eval().requires_grad_(False)

    devices = {p.device.type for p in model.parameters()}
    if devices - {"cpu"}:
        raise RuntimeError(f"Expected all parameters on CPU, found devices: {devices}")
    return model, torch_dtype


def log_model_stats(model, torch_dtype) -> None:
    import torch

    n_params = sum(p.numel() for p in model.parameters())
    dtypes = sorted({str(p.dtype).replace("torch.", "") for p in model.parameters()})
    logging.info("Parameters: %d (%.3f B)", n_params, n_params / 1e9)
    logging.info("Parameter dtypes: %s", ", ".join(dtypes))
    logging.info("Requested torch_dtype: %s", torch_dtype)
    logging.info("All parameters on device: cpu")


def run_check_only(ckpt_dir: Path) -> int:
    info = inspect_checkpoint(ckpt_dir)
    est = estimate_memory(info)
    total_ram, available = system_ram_bytes()

    logging.info("Checkpoint: %s", info.ckpt_dir)
    logging.info("config.json: WanModel model_type=%s", info.config.get("model_type"))
    logging.info("Index tensors: %d across %d shards", info.num_tensors, len(info.shards))
    logging.info("Index total_size: %.3f GB", info.index_total_bytes / 1e9)
    logging.info("Shard bytes on disk: %.3f GB", info.shard_bytes_on_disk / 1e9)
    if total_ram is not None:
        logging.info("System RAM total: %.3f GB", total_ram / 1e9)
    if available is not None:
        logging.info("System RAM available: %.3f GB", available / 1e9)
    logging.info(
        "Estimate — resident after load ~%.3f GB; conservative peak ~%.3f GB",
        est.estimated_resident_bytes / 1e9,
        est.estimated_peak_during_load_bytes / 1e9,
    )

    try:
        plan = build_loading_plan()
        logging.info("Planned load path: %s", plan.method)
        logging.info("Diffusers %s, accelerate=%s", plan.diffusers_version, plan.accelerate_available)
    except RuntimeError as exc:
        logging.error("LOAD BLOCKED: %s", exc)
        return 2

    reason = load_refusal_reason(est, total_ram, available)
    if reason:
        logging.error("LOAD REFUSED: %s", reason)
        return 2

    logging.info("Pre-flight OK — run without --check-only to attempt CPU load.")
    return 0


def run_load(ckpt_dir: Path) -> int:
    rss_before = _rss_mb()
    logging.info("Process RSS before load: %.1f MB", rss_before)

    info = inspect_checkpoint(ckpt_dir)
    est = estimate_memory(info)
    total_ram, available = system_ram_bytes()

    try:
        plan = build_loading_plan()
    except RuntimeError as exc:
        logging.error("LOAD BLOCKED: %s", exc)
        return 2

    logging.info("Loading method: %s", plan.method)
    logging.info("Diffusers %s", plan.diffusers_version)

    reason = load_refusal_reason(est, total_ram, available)
    if reason:
        logging.error("LOAD REFUSED (not attempting): %s", reason)
        return 2

    t0 = time.perf_counter()
    try:
        model, torch_dtype = load_dit_cpu(info, plan)
    except MemoryError as exc:
        logging.error("LOAD FAILED (MemoryError): %s", exc)
        logging.error("RSS at failure: %.1f MB", _rss_mb())
        return 2
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg or "alloc" in msg:
            logging.error("LOAD FAILED (likely OOM): %s", exc)
            logging.error("RSS at failure: %.1f MB", _rss_mb())
            return 2
        logging.error("LOAD FAILED: %s", exc)
        return 2
    except OSError as exc:
        logging.error("LOAD FAILED (OSError): %s", exc)
        return 2

    elapsed = time.perf_counter() - t0
    rss_after = _rss_mb()
    logging.info("Load elapsed: %.2f s", elapsed)
    logging.info("Process RSS after load: %.1f MB (delta %.1f MB)", rss_after, rss_after - rss_before)
    log_model_stats(model, torch_dtype)

    del model
    gc.collect()
    logging.info("CPU DiT load smoke test passed.")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=DEFAULT_CKPT,
        help=f"TI2V-5B checkpoint directory (default: {DEFAULT_CKPT})",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate checkpoint and memory only; do not load weights.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    ckpt = args.ckpt_dir.resolve()
    try:
        return run_check_only(ckpt) if args.check_only else run_load(ckpt)
    except (FileNotFoundError, ValueError) as exc:
        logging.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        logging.error("Interrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
