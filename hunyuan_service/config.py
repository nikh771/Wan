"""HunyuanVideo-1.5 paths and availability checks."""

from __future__ import annotations

from pathlib import Path

HUNYUAN_MODEL_ID = "HunyuanVideo-1.5"
HUNYUAN_HF_ID = "tencent/HunyuanVideo-1.5"
DEFAULT_RESOLUTION = "480p"
DEFAULT_ASPECT_RATIO = "16:9"
DEFAULT_FPS = 24


def repo_ready(repo_dir: Path) -> bool:
    return (repo_dir / "generate.py").is_file()


def model_available(model_root: Path) -> bool:
    if not model_root.is_dir():
        return False
    transformer = model_root / "transformer"
    if transformer.is_dir() and any(transformer.iterdir()):
        return True
    # Partial snapshot layout
    return (model_root / "model_index.json").is_file() or any(model_root.iterdir())