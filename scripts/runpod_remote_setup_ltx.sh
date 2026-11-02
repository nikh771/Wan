#!/usr/bin/env bash
# LTX-Video 2B setup (isolated venv_ltx). Run from repo root:
#   bash scripts/runpod_remote_setup_ltx.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
LTX_DIR="${ROOT}/third_party/LTX-Video"

echo "==> LTX engine setup | project root: ${ROOT}"

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

echo "==> API venv (shared service)..."
python3 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-storyboard.txt

echo "==> LTX inference venv (isolated from Wan)..."
python3 -m venv venv_ltx
# shellcheck disable=SC1091
source venv_ltx/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

if [[ ! -d "${LTX_DIR}/.git" ]]; then
  git clone --depth 1 https://github.com/Lightricks/LTX-Video.git "${LTX_DIR}"
fi
python -m pip install -e "${LTX_DIR}[inference]"

if [[ ! -f "${ROOT}/.env" ]]; then
  cp .env.example .env
fi
KEY_BLOCK=""
if ! grep -q '^WAN_WORKER_API_KEY=' "${ROOT}/.env" 2>/dev/null; then
  KEY="$(python -c "import secrets; print(secrets.token_urlsafe(32))")"
  KEY_BLOCK="WAN_WORKER_API_KEY=${KEY}"
fi
{
  echo ""
  echo "${KEY_BLOCK}"
  echo "LTX_PYTHON=venv_ltx/bin/python"
  echo "LTX_REPO_DIR=third_party/LTX-Video"
  echo "LTX_PIPELINE_CONFIG=configs/ltxv-2b-0.9.6-dev.yaml"
  echo "WAN_PERSISTENT_INFERENCE=false"
} >> "${ROOT}/.env"

echo "LTX setup complete. Download weights on first inference or via Hugging Face."
echo "Create jobs with engine_type=ltx"
