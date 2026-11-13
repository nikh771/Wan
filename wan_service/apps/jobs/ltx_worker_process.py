"""Deprecated shim — use ltx_service/worker_main.py via LTX_PYTHON."""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from ltx_service.worker_main import main

    raise SystemExit(main())
