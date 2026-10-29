#!/usr/bin/env bash
# One-time (or repeat) setup on a Linux CUDA host — e.g. RunPod over SSH.
# Run from the repo root on the remote machine:
#   bash scripts/runpod_remote_setup.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

echo "==> Project root: ${ROOT}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Use a GPU pod with NVIDIA drivers."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "==> Installing ffmpeg (apt)..."
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends ffmpeg
  else
    echo "WARN: ffmpeg missing and apt-get unavailable; install ffmpeg manually."
  fi
fi

if [[ ! -d "${ROOT}/venv" ]]; then
  echo "==> Creating venv..."
  python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate

python -m pip install --upgrade pip

echo "==> Installing CUDA PyTorch (cu128 index)..."
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

echo "==> Installing project requirements..."
python -m pip install -r requirements.txt

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

if [[ ! -f "${ROOT}/.env" ]]; then
  cp .env.example .env
  KEY="$(python -c "import secrets; print(secrets.token_urlsafe(32))")"
  if grep -q '^WAN_WORKER_API_KEY=$' .env 2>/dev/null || ! grep -q '^WAN_WORKER_API_KEY=' .env; then
    echo "WAN_WORKER_API_KEY=${KEY}" >> .env
  fi
  chmod 600 .env
  echo "==> Created .env from .env.example (generated WAN_WORKER_API_KEY)."
else
  echo "==> .env already exists; leaving unchanged."
fi

if [[ ! -f third_party/Wan2.2/generate.py ]]; then
  echo "ERROR: third_party/Wan2.2/generate.py missing. Clone or sync the full repo."
  exit 1
fi

CKPT="${ROOT}/models/Wan2.2-TI2V-5B"
if [[ ! -d "${CKPT}" ]] || [[ -z "$(ls -A "${CKPT}" 2>/dev/null || true)" ]]; then
  echo "==> Downloading TI2V-5B weights (~32 GB). This takes a while..."
  bash scripts/download_ti2v_5b_weights.sh
else
  echo "==> Checkpoints already present under ${CKPT}"
fi

echo ""
echo "Setup complete. Start the API:"
echo "  cd ${ROOT} && source venv/bin/activate"
echo "  uvicorn wan_service.main:app --host 0.0.0.0 --port 8080"
echo ""
echo "From your Mac (with SSH config Host runpod-wan), tunnel the port:"
echo "  ssh -N -L 8080:127.0.0.1:8080 runpod-wan"
echo "  curl http://127.0.0.1:8080/health"
