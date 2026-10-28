"""
Backward-compatible module-level settings (tests may monkeypatch these names).

Prefer ``wan_service.core.config.settings`` for new code.
"""

from __future__ import annotations

from pathlib import Path

from wan_service.core.config import get_settings, settings as _settings

_settings_ref = get_settings()

WAN_REPO_DIR: Path = _settings_ref.wan_repo_dir
WAN_CKPT_DIR: Path = _settings_ref.wan_ckpt_dir
WAN_GENERATE_SCRIPT: Path = _settings_ref.wan_generate_script
WAN_PYTHON: str = _settings_ref.WAN_PYTHON

WORKER_API_KEY: str = _settings_ref.WAN_WORKER_API_KEY
WORKER_HOST: str = _settings_ref.WAN_WORKER_HOST
WORKER_PORT: int = _settings_ref.WAN_WORKER_PORT

JOBS_DIR: Path = _settings_ref.jobs_dir
MAX_QUEUE_SIZE: int = _settings_ref.WAN_MAX_QUEUE_SIZE
MAX_IMAGE_BYTES: int = _settings_ref.WAN_MAX_IMAGE_BYTES
MAX_PROMPT_CHARS: int = _settings_ref.WAN_MAX_PROMPT_CHARS

FFMPEG_BINARY: str = _settings_ref.FFMPEG_BINARY
FFPROBE_BINARY: str = _settings_ref.FFPROBE_BINARY

SIZE_LANDSCAPE: str = _settings_ref.WAN_SIZE_LANDSCAPE
SIZE_PORTRAIT: str = _settings_ref.WAN_SIZE_PORTRAIT
