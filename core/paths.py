"""Repository and service root paths."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WAN_SERVICE_ROOT = REPO_ROOT / "wan_service"
LTX_SERVICE_ROOT = REPO_ROOT / "ltx_service"
