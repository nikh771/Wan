#!/usr/bin/env bash
# One-command GPU host setup: wipe prior model artifacts, install deps + weights for ONE engine.
#
#   bash scripts/runpod_remote_setup.sh wan        # default
#   bash scripts/runpod_remote_setup.sh ltx
#   bash scripts/runpod_remote_setup.sh cogvideox
#
# Full switch (default): removes other engines' checkpoints, all venvs, job outputs,
# pip/HF caches, then installs only the chosen engine.
#
#   GPU_SETUP_FULL_SWITCH=0 bash scripts/runpod_remote_setup.sh ltx   # keep other weights
#   GPU_SETUP_SKIP_DISK_CLEAN=1 bash scripts/runpod_remote_setup.sh wan
#   GPU_SETUP_PURGE_CLONES=1 bash scripts/runpod_remote_setup.sh ltx    # re-clone LTX-Video
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE="${1:-wan}"

export ROOT
export GPU_SETUP_FULL_SWITCH="${GPU_SETUP_FULL_SWITCH:-1}"

case "${ENGINE}" in
  wan)
    export GPU_SETUP_TARGET_ENGINE=wan
    exec bash "${ROOT}/scripts/runpod_remote_setup_wan.sh"
    ;;
  ltx)
    export GPU_SETUP_TARGET_ENGINE=ltx
    exec bash "${ROOT}/scripts/runpod_remote_setup_ltx.sh"
    ;;
  cogvideox|cogx|cog)
    export GPU_SETUP_TARGET_ENGINE=cogvideox
    exec bash "${ROOT}/scripts/runpod_remote_setup_cogvideox.sh"
    ;;
  clean)
    export GPU_SETUP_FULL_SWITCH=1
    export GPU_SETUP_TARGET_ENGINE="${2:-}"
    bash "${ROOT}/scripts/gpu_setup_clean_disk.sh"
    echo "Disk clean only. Run: bash scripts/runpod_remote_setup.sh [wan|ltx|cogvideox]"
    ;;
  *)
    echo "Usage: $0 [wan|ltx|cogvideox|clean]"
    echo ""
    echo "  wan | ltx | cogvideox   Full disk prep + venvs + model weights + .env"
    echo "  clean [engine]          Only run gpu_setup_clean_disk.sh (optional target label)"
    exit 1
    ;;
esac