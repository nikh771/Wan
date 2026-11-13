"""Minimal cross-process GPU ownership for sequential Wan / LTX / CogVideoX use.

JobManager already serializes jobs; this file records which engine last claimed
the GPU so health/benchmarks can report IDLE / COGVIDEOX_LOADED / etc.
"""

from __future__ import annotations

import json
import os
import time
from enum import Enum
from pathlib import Path
from typing import Any

from core.paths import REPO_ROOT


class GpuState(str, Enum):
    IDLE = "IDLE"
    WAN_LOADED = "WAN_LOADED"
    LTX_LOADED = "LTX_LOADED"
    COGVIDEOX_LOADED = "COGVIDEOX_LOADED"
    GENERATING = "GENERATING"


def _state_path() -> Path:
    raw = os.environ.get("WAN_JOBS_DIR", "wan_jobs").strip() or "wan_jobs"
    base = Path(raw)
    if not base.is_absolute():
        base = REPO_ROOT / base
    base.mkdir(parents=True, exist_ok=True)
    return base / ".gpu_owner_state.json"


def read_gpu_state() -> dict[str, Any]:
    path = _state_path()
    if not path.is_file():
        return {"state": GpuState.IDLE.value, "owner": None, "updated_at": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"state": GpuState.IDLE.value, "owner": None, "updated_at": None}
        return data
    except (OSError, json.JSONDecodeError):
        return {"state": GpuState.IDLE.value, "owner": None, "updated_at": None}


def write_gpu_state(state: GpuState, *, owner: str | None = None, detail: str | None = None) -> dict[str, Any]:
    payload = {
        "state": state.value if isinstance(state, GpuState) else str(state),
        "owner": owner,
        "detail": detail,
        "updated_at": time.time(),
        "pid": os.getpid(),
    }
    path = _state_path()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
