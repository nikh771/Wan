#!/usr/bin/env bash
# Default remote setup dispatches to an engine-specific script.
#   bash scripts/runpod_remote_setup.sh          # Wan (default)
#   bash scripts/runpod_remote_setup.sh wan
#   bash scripts/runpod_remote_setup.sh ltx
#   bash scripts/runpod_remote_setup.sh storyboard
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE="${1:-wan}"

case "${ENGINE}" in
  wan)
    exec bash "${ROOT}/scripts/runpod_remote_setup_wan.sh"
    ;;
  ltx)
    exec bash "${ROOT}/scripts/runpod_remote_setup_ltx.sh"
    ;;
  storyboard)
    exec bash "${ROOT}/scripts/runpod_remote_setup_storyboard.sh"
    ;;
  *)
    echo "Usage: $0 [wan|ltx|storyboard]"
    exit 1
    ;;
esac
