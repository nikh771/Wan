#!/usr/bin/env bash
# Wan 2.2 TI2V-5B GPU worker setup. Run from repo root:
#   bash scripts/runpod_remote_setup_wan.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

echo "==> Wan engine setup | project root: ${ROOT}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Use a GPU pod with NVIDIA drivers."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "==> Installing ffmpeg (apt)..."
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y --no-install-recommends ffmpeg
  else
    echo "WARN: ffmpeg missing; install ffmpeg manually."
  fi
fi

CKPT_DIR="${ROOT}/models/Wan2.2-TI2V-5B"
JOBS_DIR="${ROOT}/wan_jobs"
WS_AVAIL_GB=0
if [[ -d /workspace ]]; then
  WS_AVAIL_GB="$(df -BG /workspace 2>/dev/null | awk 'NR==2 {gsub(/G/,"",$4); print $4}' || echo 0)"
fi
if [[ "${WS_AVAIL_GB}" -ge 35 ]]; then
  CKPT_DIR="/workspace/models/Wan2.2-TI2V-5B"
  JOBS_DIR="/workspace/wan_jobs"
  echo "==> Using /workspace for weights (${WS_AVAIL_GB} GB free)"
fi

echo "==> Creating venv (Wan)..."
python3 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install --upgrade pip
pip cache purge 2>/dev/null || true

echo "==> Installing requirements-gpu.txt..."
python -m pip install -r requirements-gpu.txt
python -m pip install --no-deps --only-binary=flash-attn -r requirements-gpu-flash.txt

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda)"
python -c "import flash_attn; print('flash_attn:', flash_attn.__version__)"

if [[ ! -f third_party/Wan2.2/generate.py ]]; then
  echo "ERROR: third_party/Wan2.2/generate.py missing."
  exit 1
fi

if [[ ! -f "${ROOT}/.env" ]]; then
  cp .env.example .env
  KEY="$(python -c "import secrets; print(secrets.token_urlsafe(32))")"
  {
    echo ""
    echo "WAN_WORKER_API_KEY=${KEY}"
    echo "WAN_PYTHON=venv/bin/python"
    echo "WAN_CKPT_DIR=${CKPT_DIR}"
    echo "WAN_JOBS_DIR=${JOBS_DIR}"
  } >> .env
  chmod 600 .env
else
  echo "==> .env exists; set WAN_PYTHON=venv/bin/python WAN_CKPT_DIR=${CKPT_DIR}"
fi

export WAN_CKPT_DIR="${CKPT_DIR}"
if [[ ! -d "${CKPT_DIR}" ]] || [[ -z "$(ls -A "${CKPT_DIR}" 2>/dev/null || true)" ]]; then
  bash scripts/download_ti2v_5b_weights.sh
fi

echo "Wan setup complete. API: uvicorn wan_service.main:app --host 0.0.0.0 --port 8080"
