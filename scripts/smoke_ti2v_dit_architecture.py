#!/usr/bin/env python3
"""
Instantiate Wan 2.2 TI2V-5B DiT architecture from checkpoint config only (CPU, no weights).

WARNING: This validates architecture construction and random initialization only.
It does NOT load pretrained safetensor shards or verify weight compatibility.

  cd Wan_service
  .venv/bin/python scripts/smoke_ti2v_dit_architecture.py --plan-only
  .venv/bin/python scripts/smoke_ti2v_dit_architecture.py --ckpt-dir models/Wan2.2-TI2V-5B

Construction path: ``WanModel.load_config(ckpt_dir)`` then ``WanModel.from_config(config)``.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import logging
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
WAN_ROOT = REPO_ROOT / "third_party" / "Wan2.2"
DEFAULT_CKPT = REPO_ROOT / "models" / "Wan2.2-TI2V-5B"
INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"
OS_RESERVE_BYTES = 2 * 1024**3

# ``@register_to_config`` keys on ``WanModel.__init__`` (see wan/modules/model.py).
WAN_MODEL_INIT_PARAM_NAMES: frozenset[str] = frozenset(
    {
        "model_type",
        "patch_size",
        "text_len",
        "in_dim",
        "dim",
        "ffn_dim",
        "freq_dim",
        "text_dim",
        "out_dim",
        "num_heads",
        "num_layers",
        "window_size",
        "qk_norm",
        "cross_attn_norm",
        "eps",
    }
)

ARCHITECTURE_WARNING = (
    "ARCHITECTURE-ONLY: random CPU weights from WanModel.from_config — "
    "not pretrained checkpoint loading or inference readiness."
)


@dataclass(frozen=True)
class ConfigInfo:
    ckpt_dir: Path
    raw: dict[str, Any]
    init_kwargs_from_json: dict[str, Any]
    index_total_bytes: int | None


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


def read_checkpoint_config_json(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def validate_wan_ti2v_config(config: dict[str, Any]) -> None:
    if config.get("_class_name") != "WanModel":
        raise ValueError(
            f"config.json _class_name={config.get('_class_name')!r}, expected 'WanModel'"
        )
    if config.get("model_type") != "ti2v":
        raise ValueError(
            f"config.json model_type={config.get('model_type')!r}, expected 'ti2v'"
        )


def init_kwargs_from_checkpoint_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Map ``config.json`` fields to ``WanModel.__init__`` keyword names.

    Omits keys not present in the JSON (Diffusers ``from_config`` supplies defaults).
    """
    validate_wan_ti2v_config(config)
    return {
        key: config[key]
        for key in config
        if not key.startswith("_") and key in WAN_MODEL_INIT_PARAM_NAMES
    }


def optional_index_total_bytes(ckpt_dir: Path) -> int | None:
    index_path = ckpt_dir / INDEX_NAME
    if not index_path.is_file():
        return None
    index = json.loads(index_path.read_text(encoding="utf-8"))
    meta = index.get("metadata") or {}
    total = int(meta.get("total_size") or 0)
    return total if total > 0 else None


def inspect_config(ckpt_dir: Path) -> ConfigInfo:
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    raw = read_checkpoint_config_json(ckpt_dir / "config.json")
    validate_wan_ti2v_config(raw)
    return ConfigInfo(
        ckpt_dir=ckpt_dir,
        raw=raw,
        init_kwargs_from_json=init_kwargs_from_checkpoint_config(raw),
        index_total_bytes=optional_index_total_bytes(ckpt_dir),
    )


def estimate_random_init_bytes(info: ConfigInfo) -> int | None:
    """
    Rough resident RAM after ``from_config`` (default float32 parameters).

    Uses checkpoint index ``total_size`` when present (same order of magnitude as
    a full fp32/bf16 weight footprint); otherwise unknown.
    """
    if info.index_total_bytes is not None:
        # Random init uses default float32; index total_size for TI2V-5B is ~20 GB (≈5B × 4 B).
        return info.index_total_bytes
    return None


def memory_refusal_reason(estimated_bytes: int | None) -> str | None:
    if estimated_bytes is None:
        return None
    total_ram, available = system_ram_bytes()
    need = estimated_bytes + OS_RESERVE_BYTES
    if total_ram is not None and need > total_ram:
        return (
            f"Estimated CPU RAM for full random-init DiT (~{estimated_bytes / 1e9:.2f} GB parameters "
            f"+ ~{OS_RESERVE_BYTES / 1e9:.1f} GB reserve) exceeds system RAM ({total_ram / 1e9:.2f} GB). "
            "Use --plan-only to validate config without constructing the model, or --force on a machine with "
            "enough memory."
        )
    if available is not None and need > available:
        return (
            f"Estimated RAM need (~{need / 1e9:.2f} GB) exceeds currently available "
            f"({available / 1e9:.2f} GB). Free memory, use --plan-only, or pass --force."
        )
    return None


def ensure_wan_package_stubs() -> None:
    if str(WAN_ROOT) not in sys.path:
        sys.path.insert(0, str(WAN_ROOT))

    def _pkg(name: str, path: Path) -> ModuleType:
        if name in sys.modules:
            return sys.modules[name]
        mod = ModuleType(name)
        mod.__path__ = [str(path)]
        mod.__package__ = name
        sys.modules[name] = mod
        return mod

    _pkg("wan", WAN_ROOT / "wan")
    _pkg("wan.modules", WAN_ROOT / "wan" / "modules")


def _load_submodule(fullname: str, file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(fullname, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {fullname} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = fullname.rpartition(".")[0]
    sys.modules[fullname] = mod
    spec.loader.exec_module(mod)
    return mod


def import_wan_model_class():
    """``wan.modules.model.WanModel`` without importing ``wan/__init__.py`` (T5/VAE/einops)."""
    ensure_wan_package_stubs()
    if "wan.device_compat" not in sys.modules:
        _load_submodule("wan.device_compat", WAN_ROOT / "wan" / "device_compat.py")
    if "wan.modules.attention" not in sys.modules:
        _load_submodule(
            "wan.modules.attention", WAN_ROOT / "wan" / "modules" / "attention.py"
        )
    if "wan.modules.model" not in sys.modules:
        model_mod = _load_submodule(
            "wan.modules.model", WAN_ROOT / "wan" / "modules" / "model.py"
        )
    else:
        model_mod = sys.modules["wan.modules.model"]

    return model_mod.WanModel


def load_diffusers_config(WanModel, ckpt_dir: Path) -> dict[str, Any]:
    """``ConfigMixin.load_config`` for the checkpoint directory (config.json only)."""
    config = WanModel.load_config(str(ckpt_dir))
    if isinstance(config, tuple):
        config = config[0]
    return dict(config)


def instantiate_dit_from_config(WanModel, ckpt_dir: Path):
    import torch

    torch.set_default_device("cpu")
    config = load_diffusers_config(WanModel, ckpt_dir)
    validate_wan_ti2v_config(config)
    t0 = time.perf_counter()
    model = WanModel.from_config(config)
    elapsed = time.perf_counter() - t0
    model.eval().requires_grad_(False)

    devices = {p.device.type for p in model.parameters()}
    if devices - {"cpu"}:
        raise RuntimeError(f"Expected all parameters on CPU, found: {devices}")
    return model, elapsed


def log_model_stats(model, elapsed_s: float, rss_before: float, rss_after: float) -> None:
    n_params = sum(p.numel() for p in model.parameters())
    dtypes = sorted({str(p.dtype).replace("torch.", "") for p in model.parameters()})
    logging.info("Init duration: %.2f s", elapsed_s)
    logging.info("Process RSS: before=%.1f MB after=%.1f MB (delta=%.1f MB)", rss_before, rss_after, rss_after - rss_before)
    logging.info("Parameters: %d (%.3f B)", n_params, n_params / 1e9)
    logging.info("Parameter dtypes: %s", ", ".join(dtypes))
    logging.info("All parameters on device: cpu")


def run_plan_only(info: ConfigInfo) -> int:
    total_ram, available = system_ram_bytes()
    est = estimate_random_init_bytes(info)

    logging.warning(ARCHITECTURE_WARNING)
    logging.info("Checkpoint: %s", info.ckpt_dir)
    logging.info("config.json keys (model): %s", sorted(info.init_kwargs_from_json))
    logging.info(
        "Constructor kwargs from config.json: %s",
        json.dumps(info.init_kwargs_from_json, sort_keys=True),
    )
    logging.info(
        "Diffusers path: WanModel.load_config(ckpt_dir) -> WanModel.from_config(config)"
    )
    if info.index_total_bytes is not None:
        logging.info("Index total_size (reference): %.3f GB", info.index_total_bytes / 1e9)
    if est is not None:
        logging.info("Estimated random-init resident RAM: ~%.3f GB (+ reserve)", est / 1e9)
    if total_ram is not None:
        logging.info("System RAM total: %.3f GB", total_ram / 1e9)
    if available is not None:
        logging.info("System RAM available: %.3f GB", available / 1e9)

    reason = memory_refusal_reason(est)
    if reason:
        logging.warning("Full build would be refused: %s", reason)
    else:
        logging.info("Preflight OK — full build may proceed (still memory-heavy).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=DEFAULT_CKPT,
        help="TI2V-5B checkpoint directory containing config.json",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate config and memory estimate; do not construct WanModel",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Construct WanModel even when preflight estimates insufficient RAM",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    try:
        info = inspect_config(args.ckpt_dir.resolve())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logging.error("Config error: %s", exc)
        return 1

    if args.plan_only:
        return run_plan_only(info)

    est = estimate_random_init_bytes(info)
    reason = memory_refusal_reason(est)
    logging.warning(ARCHITECTURE_WARNING)
    if reason and not args.force:
        logging.error("BUILD REFUSED: %s", reason)
        return 2

    rss_before = _rss_mb()
    try:
        WanModel = import_wan_model_class()
        model, elapsed = instantiate_dit_from_config(WanModel, info.ckpt_dir)
    except Exception as exc:
        logging.exception("Failed to construct WanModel: %s", exc)
        return 1
    finally:
        gc.collect()

    rss_after = _rss_mb()
    log_model_stats(model, elapsed, rss_before, rss_after)
    del model
    gc.collect()
    logging.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
