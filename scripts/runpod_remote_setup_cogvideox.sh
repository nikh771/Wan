#!/usr/bin/env bash
# CogVideoX-2B — full setup in one run (prefer: bash scripts/runpod_remote_setup.sh cogvideox).
set -euo pipefail

unset HF_HUB_ENABLE_HF_TRANSFER
export HF_HUB_ENABLE_HF_TRANSFER=0
if [[ -f /etc/ssl/certs/ca-certificates.crt ]]; then
  export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
  export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
  export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export ROOT

if [[ ! -f "${ROOT}/scripts/gpu_setup_common.sh" ]]; then
  echo "ERROR: Missing ${ROOT}/scripts/gpu_setup_common.sh"
  echo "       If a previous setup deleted /workspace/Wan, re-clone or git pull the repo, then re-run:"
  echo "       bash scripts/runpod_remote_setup.sh cogvideox"
  exit 1
fi

# shellcheck source=gpu_setup_common.sh
source "${ROOT}/scripts/gpu_setup_common.sh"

export GPU_SETUP_TARGET_ENGINE="${GPU_SETUP_TARGET_ENGINE:-cogvideox}"
export GPU_SETUP_FULL_SWITCH="${GPU_SETUP_FULL_SWITCH:-1}"
bash "${ROOT}/scripts/gpu_setup_clean_disk.sh"
gpu_require_nvidia
gpu_ensure_ffmpeg

COGX_VENV="${ROOT}/cogvideox_service/engines/cogvideox-env"
MODEL_DIR="${ROOT}/cogvideox_service/models/cogvideox/CogVideoX-2b"
mkdir -p "${MODEL_DIR}" "${ROOT}/cogvideox_service/outputs" "${ROOT}/cogvideox_service/engines"

echo "==> CogVideoX-2B setup | ${ROOT}"
FREE_GB="$(gpu_free_gb "${ROOT}")"
echo "==> Free disk: ${FREE_GB} GB"

echo "==> API venv (requirements.txt)..."
gpu_recreate_venv "${ROOT}/venv"
gpu_install_requirements "${ROOT}/venv" "${ROOT}/requirements.txt"

USE_DEDICATED_ENV=1
if awk -v f="${FREE_GB}" 'BEGIN{exit !(f+0 < 18)}'; then
  echo "WARN: low disk (${FREE_GB} GB); CogVideoX needs ~18+ GB for dedicated venv + model."
fi

COGX_PYTHON_REL="cogvideox_service/engines/cogvideox-env/bin/python"
COGX_ENV_REL="cogvideox_service/engines/cogvideox-env"

if [[ "${USE_DEDICATED_ENV}" == "1" ]]; then
  echo "==> CogVideoX worker venv..."
  gpu_recreate_venv "${COGX_VENV}"
  # shellcheck disable=SC1091
  source "${COGX_VENV}/bin/activate"
  python -m pip install --upgrade pip
  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  python -m pip install -r "${ROOT}/cogvideox_service/requirements-gpu.txt"
fi

gpu_ensure_api_key
gpu_upsert_env "COGVIDEOX_PYTHON" "${COGX_PYTHON_REL}"
gpu_upsert_env "COGVIDEOX_ENV_DIR" "${COGX_ENV_REL}"
gpu_upsert_env "COGVIDEOX_MODEL_PATH" "cogvideox_service/models/cogvideox/CogVideoX-2b"
gpu_upsert_env "COGVIDEOX_HF_ID" "zai-org/CogVideoX-2b"
gpu_upsert_env "COGVIDEOX_MEMORY_MODE" "memory"
gpu_upsert_env "COGVIDEOX_DTYPE" "float16"
gpu_upsert_env "WAN_PERSISTENT_INFERENCE" "false"
gpu_upsert_env "HF_HUB_ENABLE_HF_TRANSFER" "0"

echo "==> Downloading CogVideoX-2B weights..."
COGVIDEOX_PYTHON="${COGX_PYTHON_REL}" \
COGVIDEOX_MODEL_PATH="cogvideox_service/models/cogvideox/CogVideoX-2b" \
  bash "${ROOT}/scripts/download_cogvideox_2b.sh"

gpu_print_start_api
echo "    Test:  python scripts/test_cogvideox.py"
echo "    Jobs:  engine_type=cogvideox, mode=t2v on POST /video/generate"