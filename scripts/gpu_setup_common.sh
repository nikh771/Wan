#!/usr/bin/env bash
# Shared helpers for GPU engine setup scripts (source, do not execute directly).
set -euo pipefail

gpu_setup_root() {
  if [[ -z "${ROOT:-}" ]]; then
    ROOT="$(cd "$(dirname "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")/.." && pwd)"
  fi
  export ROOT
  cd "${ROOT}"
}

gpu_require_nvidia() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. Use a GPU host with NVIDIA drivers."
    exit 1
  fi
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
}

gpu_ensure_ffmpeg() {
  if command -v ffmpeg >/dev/null 2>&1; then
    return 0
  fi
  if command -v apt-get >/dev/null 2>&1; then
    echo "==> Installing ffmpeg (+ git when apt)..."
    apt-get update -qq
    apt-get install -y --no-install-recommends ffmpeg git curl
  else
    echo "WARN: ffmpeg not found; install ffmpeg manually."
  fi
}

gpu_ensure_git() {
  if command -v git >/dev/null 2>&1; then
    return 0
  fi
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y --no-install-recommends git
  fi
}

gpu_upsert_env() {
  local key="$1"
  local value="$2"
  local env_file="${ROOT}/.env"
  if [[ ! -f "${env_file}" ]]; then
    cp "${ROOT}/.env.example" "${env_file}"
    chmod 600 "${env_file}"
  fi
  if grep -q "^${key}=" "${env_file}" 2>/dev/null; then
    sed -i "s|^${key}=.*|${key}=${value}|" "${env_file}"
  else
    echo "${key}=${value}" >> "${env_file}"
  fi
}

gpu_ensure_api_key() {
  local env_file="${ROOT}/.env"
  if [[ ! -f "${env_file}" ]]; then
    cp "${ROOT}/.env.example" "${env_file}"
    chmod 600 "${env_file}"
  fi
  if ! grep -q '^WAN_WORKER_API_KEY=.' "${env_file}" 2>/dev/null; then
    local key
    key="$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")"
    echo "WAN_WORKER_API_KEY=${key}" >> "${env_file}"
    chmod 600 "${env_file}"
  fi
}

# Recreate venv from scratch (caller sets GPU_SETUP_REBUILD_VENVS=1 via full switch).
gpu_recreate_venv() {
  local venv_path="$1"
  if [[ -d "${venv_path}" ]]; then
    chmod -R u+w "${venv_path}" 2>/dev/null || true
    rm -rf "${venv_path}"
  fi
  if ! python3 -m venv --copies "${venv_path}" 2>/dev/null; then
    rm -rf "${venv_path}"
    python3 -m venv --copies --without-pip "${venv_path}"
    # shellcheck disable=SC1091
    source "${venv_path}/bin/activate"
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    python /tmp/get-pip.py
    deactivate 2>/dev/null || true
  fi
}

gpu_install_requirements() {
  local venv_path="$1"
  local requirements_file="$2"
  # shellcheck disable=SC1091
  source "${venv_path}/bin/activate"
  python -m pip install --upgrade pip
  python -m pip install -r "${requirements_file}"
}

gpu_free_gb() {
  local path="$1"
  local free_kb
  free_kb="$(df -Pk "${path}" 2>/dev/null | awk 'NR==2 {print $4}' || echo 0)"
  awk -v k="${free_kb}" 'BEGIN{printf "%.1f", k/1024/1024}'
}

# Print post-setup instructions.
gpu_print_start_api() {
  echo ""
  echo "==> Setup complete. Start the API (from repo root):"
  echo "    source venv/bin/activate"
  echo "    uvicorn wan_service.main:app --host 0.0.0.0 --port 8080"
  echo "    Health: curl -s http://127.0.0.1:8080/health | python -m json.tool"
}