#!/usr/bin/env bash
# CogVideoX-2B setup (isolated cogvideox_service/engines/cogvideox-env).
# Run from repo root:
#   bash scripts/runpod_remote_setup_cogvideox.sh
#
# Same pattern as LTX:
#   - separate package: cogvideox_service/
#   - separate Python env: cogvideox_service/engines/cogvideox-env
#   - separate model dir:  cogvideox_service/models/cogvideox/CogVideoX-2b
#   - API never loads weights; jobs run via COGVIDEOX_PYTHON worker_main.py
set -euo pipefail

unset HF_HUB_ENABLE_HF_TRANSFER
export HF_HUB_ENABLE_HF_TRANSFER=0
# Avoid flaky pip TLS when vendor cacert.pem is mid-replace during upgrades.
if [[ -f /etc/ssl/certs/ca-certificates.crt ]]; then
  export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
  export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
  export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
COGX_VENV="${ROOT}/cogvideox_service/engines/cogvideox-env"
MODEL_DIR="${ROOT}/cogvideox_service/models/cogvideox/CogVideoX-2b"
LTX_VENV="${ROOT}/ltx_service/engines/ltx-env"

echo "==> CogVideoX-2B separate service setup | ${ROOT}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found."
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y --no-install-recommends ffmpeg git
  fi
fi

mkdir -p "${MODEL_DIR}"
mkdir -p "${ROOT}/cogvideox_service/outputs"
mkdir -p "${ROOT}/cogvideox_service/engines"

FREE_KB="$(df -Pk "${ROOT}" | awk 'NR==2{print $4}')"
FREE_GB="$(awk -v k="${FREE_KB}" 'BEGIN{printf "%.1f", k/1024/1024}')"
echo "==> Free disk on workspace: ${FREE_GB} GB"

# Prefer a dedicated CogVideoX venv. If disk is too tight for a second CUDA
# torch install (~5–7GB) + model (~13GB), reuse the existing LTX env Python
# (compatible torch/diffusers) but keep models under cogvideox_service/.
USE_DEDICATED_ENV=1
NEED_GB=20
if awk -v f="${FREE_GB}" -v n="${NEED_GB}" 'BEGIN{exit !(f+0 < n)}'; then
  echo "WARN: only ${FREE_GB} GB free; dedicated CUDA venv+model needs ~${NEED_GB} GB."
  if [[ -x "${LTX_VENV}/bin/python" ]]; then
    echo "WARN: reusing LTX isolated Python for CogVideoX worker (models stay separate)."
    USE_DEDICATED_ENV=0
  fi
fi

if [[ ! -f "${ROOT}/.env" ]]; then
  cp .env.example .env
fi

upsert_env() {
  local key="$1"
  local value="$2"
  if grep -q "^${key}=" "${ROOT}/.env" 2>/dev/null; then
    sed -i "s|^${key}=.*|${key}=${value}|" "${ROOT}/.env"
  else
    echo "${key}=${value}" >> "${ROOT}/.env"
  fi
}

if [[ "${USE_DEDICATED_ENV}" == "1" ]]; then
  echo "==> Creating isolated CogVideoX env at cogvideox_service/engines/cogvideox-env ..."
  # Drop broken/partial envs (interrupted installs break ensurepip on recreate).
  if [[ -e "${COGX_VENV}" ]]; then
    echo "==> Removing existing ${COGX_VENV} ..."
    chmod -R u+w "${COGX_VENV}" 2>/dev/null || true
    rm -rf "${COGX_VENV}"
  fi
  # --copies avoids overlay/symlink breakage seen with ensurepip on this host.
  if ! python3 -m venv --copies "${COGX_VENV}"; then
    echo "WARN: venv+ensurepip failed; retrying with --without-pip + get-pip.py"
    rm -rf "${COGX_VENV}"
    python3 -m venv --copies --without-pip "${COGX_VENV}"
    # shellcheck disable=SC1091
    source "${COGX_VENV}/bin/activate"
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    python /tmp/get-pip.py
  else
    # shellcheck disable=SC1091
    source "${COGX_VENV}/bin/activate"
  fi
  python -m pip install --upgrade pip
  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  python -m pip install -r "${ROOT}/cogvideox_service/requirements-gpu.txt"
  COGX_PYTHON_REL="cogvideox_service/engines/cogvideox-env/bin/python"
  COGX_ENV_REL="cogvideox_service/engines/cogvideox-env"
else
  # Ensure CogVideoX-specific pip deps exist in the shared LTX worker env.
  # shellcheck disable=SC1091
  source "${LTX_VENV}/bin/activate"
  python -m pip install -r "${ROOT}/cogvideox_service/requirements-gpu.txt"
  COGX_PYTHON_REL="ltx_service/engines/ltx-env/bin/python"
  COGX_ENV_REL="ltx_service/engines/ltx-env"
  echo "NOTE: dedicated cogvideox-env skipped due to disk; COGVIDEOX_PYTHON → LTX env."
fi

upsert_env "COGVIDEOX_PYTHON" "${COGX_PYTHON_REL}"
upsert_env "COGVIDEOX_ENV_DIR" "${COGX_ENV_REL}"
upsert_env "COGVIDEOX_MODEL_PATH" "cogvideox_service/models/cogvideox/CogVideoX-2b"
upsert_env "COGVIDEOX_HF_ID" "zai-org/CogVideoX-2b"
upsert_env "COGVIDEOX_MEMORY_MODE" "memory"
upsert_env "COGVIDEOX_DTYPE" "float16"
upsert_env "WAN_PERSISTENT_INFERENCE" "false"
upsert_env "HF_HUB_ENABLE_HF_TRANSFER" "0"

if [[ ! -f "${MODEL_DIR}/model_index.json" ]]; then
  echo "==> Downloading CogVideoX-2B weights (explicit; not at API startup)..."
  COGVIDEOX_PYTHON="${COGX_PYTHON_REL}" \
  COGVIDEOX_MODEL_PATH="cogvideox_service/models/cogvideox/CogVideoX-2b" \
    bash "${ROOT}/scripts/download_cogvideox_2b.sh"
else
  echo "==> Model already present: ${MODEL_DIR}"
fi

echo "CogVideoX separate service setup complete."
echo "  Package:  cogvideox_service/"
echo "  Python:   ${COGX_PYTHON_REL}"
echo "  Model:    cogvideox_service/models/cogvideox/CogVideoX-2b"
echo "  Test:     python scripts/test_cogvideox.py"
echo "  Health:   GET /api/v1/video/engines/cogvideox/health"
echo "  Jobs:     engine_type=cogvideox (same /jobs API as wan/ltx)"
