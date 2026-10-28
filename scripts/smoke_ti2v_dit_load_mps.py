#!/usr/bin/env python3
"""
Memory-conscious DiT-only load smoke test for Wan 2.2 TI2V-5B on Apple MPS hosts.

Does not load T5/VAE, does not denoise, does not move the full model to MPS.
Uses the official ``WanModel.from_pretrained(checkpoint_dir)`` path (Diffusers).

Run explicitly (not part of pytest):

  cd /path/to/Wan_service
  export WAN_INFERENCE_BACKEND=mac_mps
  .venv/bin/python scripts/smoke_ti2v_dit_load_mps.py \\
    --ckpt-dir models/Wan2.2-TI2V-5B

Check shards and memory budget without loading weights:

  .venv/bin/python scripts/smoke_ti2v_dit_load_mps.py --check-only

Success (full run):
  - All DiT safetensor shards present
  - ``WanModel.from_pretrained`` completes with weights on CPU
  - Process exits 0 and logs parameter count + RSS

Failure / blocker:
  - Missing shards, missing diffusers, insufficient RAM vs checkpoint metadata,
    or ``MemoryError`` / MPS OOM during load → exit 2 with explanation (no retries)
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import resource
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WAN_ROOT = REPO_ROOT / "third_party" / "Wan2.2"
DEFAULT_CKPT = REPO_ROOT / "models" / "Wan2.2-TI2V-5B"
INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"
# Unified-memory headroom: OS + load buffers beyond raw shard bytes on disk.
RAM_HEADROOM_RATIO = 1.12


def _rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def _system_ram_bytes() -> tuple[int | None, int | None]:
    """Return (total_bytes, available_bytes) best-effort."""
    try:
        import psutil  # type: ignore[import-not-found]

        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available)
    except ImportError:
        pass
    try:
        total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None, None
    return total, None


def dit_index_path(ckpt_dir: Path) -> Path:
    return ckpt_dir / INDEX_NAME


def dit_shard_paths(ckpt_dir: Path) -> tuple[list[Path], int]:
    """Shard files required for DiT and total byte size from index metadata."""
    index_path = dit_index_path(ckpt_dir)
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing DiT index: {index_path}")
    data = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = data.get("weight_map") or {}
    shard_names = sorted(set(weight_map.values()))
    shards = [ckpt_dir / name for name in shard_names]
    meta = data.get("metadata") or {}
    total_size = int(meta.get("total_size") or 0)
    if total_size <= 0:
        total_size = sum(p.stat().st_size for p in shards if p.is_file())
    return shards, total_size


def validate_dit_checkpoint(ckpt_dir: Path) -> tuple[list[Path], int]:
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    config = ckpt_dir / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"Missing {config}")
    shards, total_bytes = dit_shard_paths(ckpt_dir)
    missing = [p for p in shards if not p.is_file()]
    if missing:
        names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(
            f"DiT weight shard(s) missing under {ckpt_dir}: {names}. "
            "Finish `hf download Wan-AI/Wan2.2-TI2V-5B` before running this smoke test."
        )
    on_disk = sum(p.stat().st_size for p in shards)
    return shards, max(total_bytes, on_disk)


def memory_blocker(total_weight_bytes: int) -> str | None:
    total_ram, available = _system_ram_bytes()
    need = int(total_weight_bytes * RAM_HEADROOM_RATIO)
    if total_ram is None and total_weight_bytes > 14 * 1024**3:
        return (
            f"DiT checkpoint is ~{total_weight_bytes / 1e9:.2f} GB; on a 16 GB unified-memory "
            "Mac this exceeds safe CPU load capacity for upstream WanModel.from_pretrained. "
            "Could not read system RAM (install psutil for a precise check). "
            "Do not force a full load on this host without more memory or a sharded loading plan."
        )
    if total_ram is not None and need > total_ram:
        return (
            f"DiT checkpoint metadata/disk size (~{total_weight_bytes / 1e9:.2f} GB) "
            f"exceeds system RAM ({total_ram / 1e9:.2f} GB) with "
            f"{RAM_HEADROOM_RATIO:.0%} headroom. Loading the full DiT on this host is "
            "not safe; use a machine with more unified memory or wait for a sharded "
            "load strategy (not implemented in upstream WanModel.from_pretrained)."
        )
    if available is not None and need > available:
        return (
            f"Available RAM ({available / 1e9:.2f} GB) is below estimated need "
            f"(~{need / 1e9:.2f} GB for DiT load). Free memory or use --check-only."
        )
    return None


def load_dit_on_cpu(ckpt_dir: Path):
    """Official Wan TI2V DiT load; weights remain on CPU."""
    sys.path.insert(0, str(WAN_ROOT))
    os.environ.setdefault("WAN_INFERENCE_BACKEND", "mac_mps")

    import torch

    try:
        from wan.device_compat import param_dtype_for_backend
        from wan.modules.model import WanModel
    except ImportError as exc:
        raise RuntimeError(
            "Install MPS inference deps: pip install -r requirements-mps-inference.txt"
        ) from exc

    dtype = param_dtype_for_backend(torch.bfloat16)
    logging.info("Loading WanModel.from_pretrained (CPU, low_cpu_mem_usage, dtype=%s)", dtype)
    model = WanModel.from_pretrained(
        str(ckpt_dir),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval().requires_grad_(False)
    # Explicit CPU placement; smoke test does not promote the full DiT to MPS.
    model.to(torch.device("cpu"))
    return model


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
    )


def run_check_only(ckpt_dir: Path) -> int:
    shards, total_bytes = validate_dit_checkpoint(ckpt_dir)
    total_ram, available = _system_ram_bytes()
    logging.info("Checkpoint: %s", ckpt_dir)
    logging.info("DiT shards (%d): %s", len(shards), ", ".join(p.name for p in shards))
    logging.info("DiT weight bytes (index/disk): %.2f GB", total_bytes / 1e9)
    if total_ram is not None:
        logging.info("System RAM total: %.2f GB", total_ram / 1e9)
    if available is not None:
        logging.info("System RAM available: %.2f GB", available / 1e9)
    blocker = memory_blocker(total_bytes)
    if blocker:
        logging.error("BLOCKER: %s", blocker)
        return 2
    logging.info("Pre-flight OK — safe to attempt DiT load (run without --check-only).")
    return 0


def run_load_smoke(ckpt_dir: Path) -> int:
    shards, total_bytes = validate_dit_checkpoint(ckpt_dir)
    logging.info("RSS before load: %.1f MB", _rss_mb())
    logging.info("DiT shards: %d, ~%.2f GB on disk/index", len(shards), total_bytes / 1e9)

    blocker = memory_blocker(total_bytes)
    if blocker:
        logging.error("BLOCKER (not attempting load): %s", blocker)
        return 2

    try:
        model = load_dit_on_cpu(ckpt_dir)
    except MemoryError as exc:
        logging.error("BLOCKER: DiT load ran out of memory: %s", exc)
        return 2
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg or "mps" in msg and "alloc" in msg:
            logging.error("BLOCKER: DiT load failed (memory): %s", exc)
            return 2
        raise

    n_params = sum(p.numel() for p in model.parameters())
    cpu_params = sum(p.numel() for p in model.parameters() if p.device.type == "cpu")
    logging.info("DiT loaded: %d parameters (%.2f B), %d on CPU", n_params, n_params / 1e9, cpu_params)
    logging.info("RSS after load: %.1f MB", _rss_mb())

    del model
    gc.collect()
    logging.info("Smoke test passed — DiT weights load on CPU via official from_pretrained.")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=DEFAULT_CKPT,
        help=f"TI2V-5B checkpoint root (default: {DEFAULT_CKPT})",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate shards and RAM budget only; do not load weights.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _configure_logging(args.verbose)

    import torch

    if os.environ.get("WAN_INFERENCE_BACKEND", "").lower() not in ("", "mac_mps"):
        logging.warning(
            "WAN_INFERENCE_BACKEND=%s; this smoke test targets mac_mps.",
            os.environ.get("WAN_INFERENCE_BACKEND"),
        )
    if not torch.backends.mps.is_available():
        logging.warning("MPS not available — load still runs on CPU-only path.")

    try:
        if args.check_only:
            return run_check_only(args.ckpt_dir.resolve())
        return run_load_smoke(args.ckpt_dir.resolve())
    except FileNotFoundError as exc:
        logging.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        logging.error("Interrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
