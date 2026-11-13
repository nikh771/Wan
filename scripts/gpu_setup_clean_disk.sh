#!/usr/bin/env bash
# Clear pip / Hugging Face / torch caches and free apt disk before GPU setup installs.
# Run automatically from runpod_remote_setup*.sh (skip with GPU_SETUP_SKIP_DISK_CLEAN=1).
#
#   bash scripts/gpu_setup_clean_disk.sh
#
# Optional:
#   GPU_SETUP_REBUILD_VENVS=1   remove repo venv + engine venvs (fresh pip installs)
#   GPU_SETUP_CLEAN_AGGRESSIVE=1  also prune stale __pycache__ under repo (not weights)
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if [[ "${GPU_SETUP_SKIP_DISK_CLEAN:-0}" == "1" ]]; then
  echo "==> Disk/cache cleanup skipped (GPU_SETUP_SKIP_DISK_CLEAN=1)"
  exit 0
fi

_log_df() {
  local label="$1"
  local path="$2"
  if [[ -d "${path}" ]] || [[ -e "${path}" ]]; then
    echo "    ${label}:"
    df -h "${path}" 2>/dev/null | awk 'NR==1 || NR==2 {print "      " $0}' || true
  fi
}

_bytes_human() {
  local b="$1"
  if [[ "${b}" -lt 1024 ]]; then
    echo "${b}B"
  elif [[ "${b}" -lt 1048576 ]]; then
    awk -v x="${b}" 'BEGIN{printf "%.1f MB", x/1024/1024}'
  else
    awk -v x="${b}" 'BEGIN{printf "%.1f GB", x/1024/1024/1024}'
  fi
}

_dir_size() {
  local d="$1"
  if [[ -d "${d}" ]]; then
    du -sk "${d}" 2>/dev/null | awk '{print $1 * 1024}' || echo 0
  else
    echo 0
  fi
}

echo "==> Pre-install disk & cache cleanup (GPU host)"
_log_df "Project" "${ROOT}"
_log_df "Workspace" "/workspace"

remove_if_dir() {
  local d="$1"
  if [[ -z "${d}" || "${d}" == "/" ]]; then
    return 0
  fi
  if [[ -d "${d}" ]]; then
    local sz
    sz="$(_dir_size "${d}")"
    echo "    removing $(_bytes_human "${sz}") from ${d}"
    chmod -R u+w "${d}" 2>/dev/null || true
    rm -rf "${d}"
  fi
}

# pip (user + any active interpreter)
if command -v python3 >/dev/null 2>&1; then
  python3 -m pip cache purge 2>/dev/null || true
fi
remove_if_dir "${HOME}/.cache/pip"

# Hugging Face / Diffusers / torch download caches (not repo model dirs)
remove_if_dir "${HOME}/.cache/huggingface"
remove_if_dir "${HF_HOME:-}"
remove_if_dir "${HUGGINGFACE_HUB_CACHE:-}"
remove_if_dir "${TRANSFORMERS_CACHE:-}"
remove_if_dir "${DIFFUSERS_CACHE:-}"
remove_if_dir "${XDG_CACHE_HOME:-${HOME}/.cache}/huggingface"
remove_if_dir "${HOME}/.cache/torch"
remove_if_dir "${HOME}/.cache/torch_extensions"
remove_if_dir "${HOME}/.cache/nv"
remove_if_dir "${HOME}/.triton"

# Stray incomplete downloads under user cache
find "${HOME}/.cache" -name '*.incomplete' -delete 2>/dev/null || true

# Common temp artifacts from interrupted installs
rm -rf /tmp/pip-* /tmp/torch_* /tmp/get-pip.py 2>/dev/null || true

# Debian/Ubuntu (RunPod images)
if command -v apt-get >/dev/null 2>&1; then
  echo "    apt-get clean / autoremove"
  apt-get clean -qq 2>/dev/null || true
  DEBIAN_FRONTEND=noninteractive apt-get autoremove -y -qq 2>/dev/null || true
fi

if [[ "${GPU_SETUP_REBUILD_VENVS:-0}" == "1" ]]; then
  echo "==> Removing Python venvs for clean reinstall (GPU_SETUP_REBUILD_VENVS=1)"
  remove_if_dir "${ROOT}/venv"
  remove_if_dir "${ROOT}/ltx_service/engines/ltx-env"
  remove_if_dir "${ROOT}/cogvideox_service/engines/cogvideox-env"
fi

if [[ "${GPU_SETUP_CLEAN_AGGRESSIVE:-0}" == "1" ]]; then
  echo "==> Aggressive: pruning __pycache__ under repo (GPU_SETUP_CLEAN_AGGRESSIVE=1)"
  find "${ROOT}" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
  find "${ROOT}" -type f -name '*.pyc' -delete 2>/dev/null || true
fi

echo "==> Disk after cleanup:"
_log_df "Project" "${ROOT}"
_log_df "Workspace" "/workspace"
echo "==> Cache cleanup done"
