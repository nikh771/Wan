#!/usr/bin/env bash
# Linux CUDA host setup (RunPod / similar). Run from repo root:
#   bash scripts/runpod_remote_setup.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

echo "==> Project root: ${ROOT}"
df -h / "${ROOT}" 2>/dev/null || df -h /

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
else
  AVAIL_GB="$(df -BG "${ROOT}" 2>/dev/null | awk 'NR==2 {gsub(/G/,"",$4); print $4}' || echo 0)"
  if [[ "${AVAIL_GB}" -lt 35 ]]; then
    echo "WARN: ~${AVAIL_GB} GB free on container disk; model needs ~32 GB. Use a 40GB+ /workspace volume or expand disk."
  fi
fi

echo "==> Creating venv..."
python3 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate

python -m pip install --upgrade pip
pip cache purge 2>/dev/null || true

echo "==> Installing GPU dependencies (requirements-gpu.txt)..."
python -m pip install -r requirements-gpu.txt

echo "==> Installing flash-attn wheel (--no-deps so torch is not upgraded)..."
python -m pip install --no-deps --only-binary=flash-attn -r requirements-gpu-flash.txt

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
python -c "import flash_attn; print('flash_attn:', flash_attn.__version__)"

if [[ ! -f third_party/Wan2.2/generate.py ]]; then
  echo "ERROR: third_party/Wan2.2/generate.py missing. Sync the full repo."
  exit 1
fi

if [[ ! -f "${ROOT}/.env" ]]; then
  cp .env.example .env
  KEY="$(python -c "import secrets; print(secrets.token_urlsafe(32))")"
  {
    echo ""
    echo "WAN_WORKER_API_KEY=${KEY}"
    echo "WAN_CKPT_DIR=${CKPT_DIR}"
    echo "WAN_JOBS_DIR=${JOBS_DIR}"
  } >> .env
  chmod 600 .env
  echo "==> Created .env (API key, WAN_CKPT_DIR, WAN_JOBS_DIR)."
else
  echo "==> .env exists; ensure WAN_CKPT_DIR=${CKPT_DIR} and WAN_JOBS_DIR=${JOBS_DIR}"
fi

export WAN_CKPT_DIR="${CKPT_DIR}"
if [[ ! -d "${CKPT_DIR}" ]] || [[ -z "$(ls -A "${CKPT_DIR}" 2>/dev/null || true)" ]]; then
  echo "==> Downloading TI2V-5B weights (~32 GB)..."
  unset HF_HUB_ENABLE_HF_TRANSFER
  bash scripts/download_ti2v_5b_weights.sh
else
  echo "==> Checkpoints already present under ${CKPT_DIR}"
fi

echo ""
echo "Setup complete. Start the API:"
echo "  cd ${ROOT} && source venv/bin/activate"
echo "  uvicorn wan_service.main:app --host 0.0.0.0 --port 8080"
echo ""
echo "Health check (on pod or via SSH tunnel):"
echo "  curl -s http://127.0.0.1:8080/health | python -m json.tool"
