#!/usr/bin/env bash
# LTX-Video 2B distilled setup (isolated ltx_service/engines/ltx-env). Run from repo root:
#   bash scripts/runpod_remote_setup_ltx.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
LTX_DIR="${ROOT}/ltx_service/third_party/LTX-Video"
LTX_VENV="${ROOT}/ltx_service/engines/ltx-env"
MODEL_DIR="${ROOT}/ltx_service/models/ltx"

echo "==> LTX 2B distilled setup | ${ROOT}"

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

echo "==> API venv (lightweight; does not install LTX torch stack)..."
python3 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-storyboard.txt

echo "==> LTX isolated env at ltx_service/engines/ltx-env ..."
python3 -m venv "${LTX_VENV}"
# shellcheck disable=SC1091
source "${LTX_VENV}/bin/activate"
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

if [[ ! -d "${LTX_DIR}/.git" ]]; then
  git clone --depth 1 https://github.com/Lightricks/LTX-Video.git "${LTX_DIR}"
fi
python -m pip install -e "${LTX_DIR}[inference]"

if [[ ! -f "${MODEL_DIR}/ltxv-2b-0.9.6-distilled-04-25.safetensors" ]]; then
  echo "==> Downloading distilled checkpoint..."
  bash "${ROOT}/scripts/download_ltx_2b_distilled.sh"
fi

if [[ ! -f "${ROOT}/.env" ]]; then
  cp .env.example .env
fi
if ! grep -q '^WAN_WORKER_API_KEY=' "${ROOT}/.env" 2>/dev/null; then
  KEY="$(python -c "import secrets; print(secrets.token_urlsafe(32))")"
  echo "WAN_WORKER_API_KEY=${KEY}" >> "${ROOT}/.env"
fi
{
  echo ""
  echo "LTX_PYTHON=ltx_service/engines/ltx-env/bin/python"
  echo "LTX_ENV_DIR=ltx_service/engines/ltx-env"
  echo "LTX_REPO_DIR=ltx_service/third_party/LTX-Video"
  echo "LTX_MODEL_PATH=ltx_service/models/ltx/ltxv-2b-0.9.6-distilled-04-25.safetensors"
  echo "LTX_PIPELINE_CONFIG=configs/ltxv-2b-0.9.6-distilled.yaml"
  echo "WAN_PERSISTENT_INFERENCE=false"
} >> "${ROOT}/.env"

echo "LTX setup complete. Test: python scripts/test_ltx.py --prompt '...' --duration 5"
echo "Health: GET /api/v1/video/engines/ltx/health"
