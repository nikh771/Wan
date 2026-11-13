"""Unified settings (shared app + Wan). LTX: ``ltx_service.settings``."""

from __future__ import annotations

from functools import lru_cache

from core.app_settings import AppSettings
from core.paths import REPO_ROOT
from wan_service.settings import RuntimeMode, WanSettings

# Re-export for runtime.py and callers.
__all__ = [
    "Settings",
    "RuntimeMode",
    "PROJECT_ROOT",
    "get_settings",
    "clear_settings_cache",
    "settings",
]

PROJECT_ROOT = REPO_ROOT


class Settings(AppSettings, WanSettings):
    """Single object for FastAPI + Wan inference (loads repo ``.env`` once)."""


@lru_cache
def get_settings() -> Settings:
    return Settings()


def clear_settings_cache() -> None:
    get_settings.cache_clear()


settings = get_settings()
