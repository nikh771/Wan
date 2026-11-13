"""Remove intermediate chunk/bridge files after successful concat."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def cleanup_generation_artifacts(
    chunk_paths: list[Path],
    extra_paths: list[Path] | None = None,
) -> list[str]:
    """Delete chunk MP4s and bridge/temp files; return basenames removed."""
    removed: list[str] = []
    for path in chunk_paths + (extra_paths or []):
        try:
            if path.is_file():
                path.unlink()
                removed.append(path.name)
        except OSError as exc:
            logger.warning("Could not delete chunk artifact %s: %s", path, exc)
    if removed:
        logger.info("Removed %d intermediate generation file(s)", len(removed))
    return removed
