#!/usr/bin/env bash
# Disk cleanup before GPU setup installs.
#
# Default when called from runpod_remote_setup_*.sh (model switch):
#   GPU_SETUP_FULL_SWITCH=1 + GPU_SETUP_TARGET_ENGINE=wan|ltx|cogvideox
#   → removes other engines' weights, all venvs, job outputs, then pip/HF caches.
#
#   bash scripts/runpod_remote_setup.sh wan     # one command: clean + install Wan
#
# Skip:
#   GPU_SETUP_SKIP_DISK_CLEAN=1
# Keep other engines' checkpoints (only refresh venvs + caches):
#   GPU_SETUP_FULL_SWITCH=0
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TARGET="${GPU_SETUP_TARGET_ENGINE:-}"
FULL_SWITCH="${GPU_SETUP_FULL_SWITCH:-0}"

if [[ "${GPU_SETUP_SKIP_DISK_CLEAN:-0}" == "1" ]]; then
  echo "==> Disk cleanup skipped (GPU_SETUP_SKIP_DISK_CLEAN=1)"
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

remove_if_dir() {
  local d="$1"
  if [[ -z "${d}" || "${d}" == "/" || "${d}" == "${ROOT}" ]]; then
    return 0
  fi
  case "${d}" in
    "${ROOT}/"*) ;;
    *) return 0 ;;
  esac
  if [[ -d "${d}" ]]; then
    local sz
    sz="$(_dir_size "${d}")"
    echo "    removing $(_bytes_human "${sz}") from ${d}"
    chmod -R u+w "${d}" 2>/dev/null || true
    rm -rf "${d}"
  fi
}

remove_path_any() {
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

# Never delete the git checkout or any path that contains ROOT (e.g. repo at /workspace/Wan).
safe_remove_external_path() {
  local d="$1"
  local norm_d norm_root
  if [[ -z "${d}" ]]; then
    return 0
  fi
  norm_d="$(cd "${d}" 2>/dev/null && pwd -P || echo "${d%/}")"
  norm_root="$(cd "${ROOT}" 2>/dev/null && pwd -P || echo "${ROOT%/}")"
  if [[ "${norm_d}" == "${norm_root}" ]]; then
    echo "    skip ${d} (same as repo ROOT — never delete the checkout)"
    return 0
  fi
  case "${norm_root}" in
    "${norm_d}"|"${norm_d}"/*)
      echo "    skip ${d} (repo ROOT is inside ${d})"
      return 0
    ;;
  esac
  remove_path_any "${d}"
}

if [[ ! -f "${ROOT}/scripts/runpod_remote_setup.sh" ]] || [[ ! -f "${ROOT}/scripts/gpu_setup_common.sh" ]]; then
  echo "ERROR: ROOT=${ROOT} is not a complete Wan repo (missing scripts/)."
  echo "       Do not point ROOT at a parent path you intend to wipe. Clone/pull the repo first."
  exit 1
fi

echo "==> Disk cleanup | ROOT=${ROOT} target=${TARGET:-none} full_switch=${FULL_SWITCH}"
_log_df "Project" "${ROOT}"
_log_df "Workspace" "/workspace"

if [[ "${FULL_SWITCH}" == "1" ]]; then
  echo "==> Full model switch cleanup (free disk for ${TARGET:-next install})"

  # Job outputs / scratch
  remove_if_dir "${ROOT}/wan_jobs"
  remove_if_dir "${ROOT}/ltx_service/outputs"
  remove_if_dir "${ROOT}/cogvideox_service/outputs"

  # Python envs (recreated by setup script)
  remove_if_dir "${ROOT}/venv"
  remove_if_dir "${ROOT}/ltx_service/engines/ltx-env"
  remove_if_dir "${ROOT}/cogvideox_service/engines/cogvideox-env"

  # All local checkpoints (setup re-downloads the target engine only)
  remove_if_dir "${ROOT}/wan_service/models/Wan2.2-TI2V-5B"
  remove_if_dir "${ROOT}/ltx_service/models/ltx"
  remove_if_dir "${ROOT}/cogvideox_service/models/cogvideox"

  # External RunPod paths only (never remove /workspace/Wan when the repo lives there).
  safe_remove_external_path "/workspace/wan_jobs"
  if [[ -d "/workspace/Wan" ]]; then
    ext_wan_root="$(cd /workspace/Wan 2>/dev/null && pwd -P || echo /workspace/Wan)"
    norm_root="$(cd "${ROOT}" && pwd -P)"
    if [[ "${ext_wan_root}" != "${norm_root}" ]] && [[ "${norm_root}" != "${ext_wan_root}"/* ]]; then
      safe_remove_external_path "/workspace/Wan"
    fi
  fi

  if [[ "${GPU_SETUP_PURGE_CLONES:-0}" == "1" ]]; then
    remove_if_dir "${ROOT}/ltx_service/third_party/LTX-Video"
  fi

  export GPU_SETUP_REBUILD_VENVS=1
fi

if [[ "${GPU_SETUP_REBUILD_VENVS:-0}" == "1" ]] && [[ "${FULL_SWITCH}" != "1" ]]; then
  echo "==> Removing Python venvs (GPU_SETUP_REBUILD_VENVS=1)"
  remove_if_dir "${ROOT}/venv"
  remove_if_dir "${ROOT}/ltx_service/engines/ltx-env"
  remove_if_dir "${ROOT}/cogvideox_service/engines/cogvideox-env"
fi

echo "==> Clearing pip / Hugging Face / torch caches"
if command -v python3 >/dev/null 2>&1; then
  python3 -m pip cache purge 2>/dev/null || true
fi
remove_path_any "${HOME}/.cache/pip"
remove_path_any "${HOME}/.cache/huggingface"
remove_path_any "${HF_HOME:-}"
remove_path_any "${HUGGINGFACE_HUB_CACHE:-}"
remove_path_any "${TRANSFORMERS_CACHE:-}"
remove_path_any "${DIFFUSERS_CACHE:-}"
remove_path_any "${XDG_CACHE_HOME:-${HOME}/.cache}/huggingface"
remove_path_any "${HOME}/.cache/torch"
remove_path_any "${HOME}/.cache/torch_extensions"
remove_path_any "${HOME}/.cache/nv"
remove_path_any "${HOME}/.triton"

find "${HOME}/.cache" -name '*.incomplete' -delete 2>/dev/null || true
rm -rf /tmp/pip-* /tmp/torch_* /tmp/get-pip.py 2>/dev/null || true

if command -v apt-get >/dev/null 2>&1; then
  echo "    apt-get clean / autoremove"
  apt-get clean -qq 2>/dev/null || true
  DEBIAN_FRONTEND=noninteractive apt-get autoremove -y -qq 2>/dev/null || true
fi

if [[ "${GPU_SETUP_CLEAN_AGGRESSIVE:-0}" == "1" ]]; then
  echo "==> Aggressive: pruning __pycache__ under repo"
  find "${ROOT}" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
  find "${ROOT}" -type f -name '*.pyc' -delete 2>/dev/null || true
fi

echo "==> Disk after cleanup:"
_log_df "Project" "${ROOT}"
_log_df "Workspace" "/workspace"
echo "==> Cleanup done"