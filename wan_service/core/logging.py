"""Backward-compatible logging — prefer ``core.logging``."""

from core.logging import configure_logging, get_logger

__all__ = ["configure_logging", "get_logger"]
