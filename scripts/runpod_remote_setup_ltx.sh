#!/usr/bin/env bash
# LTX-Video 2B distilled setup (isolated ltx_service/engines/ltx-env). Run from repo root:
#   bash scripts/runpod_remote_setup_ltx.sh
set -euo pipefail

# Avoid empty/partial HF downloads when hf_transfer is enabled but not installed.
unset HF_HUB_ENABLE_HF_TRANSFER
export HF_HUB_ENABLE_HF_TRANSFER=0

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export ROOT
bash "${ROOT}/scripts/gpu_setup_clean_disk.sh"

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
python -m pip install -r requirements.txt

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

# LTX loads T5 from PixArt-alpha/PixArt-XL-2-1024-MS at job time; prefetch now so
# first jobs do not fail on incomplete/zero-byte HF cache entries.
echo "==> Prefetching PixArt T5 text_encoder + tokenizer..."
LTX_PYTHON="${LTX_VENV}/bin/python" bash "${ROOT}/scripts/download_ltx_pixart_text_encoder.sh"

if [[ ! -f "${ROOT}/.env" ]]; then
  cp .env.example .env
fi
if ! grep -q '^WAN_WORKER_API_KEY=' "${ROOT}/.env" 2>/dev/null; then
  KEY="$(python -c "import secrets; print(secrets.token_urlsafe(32))")"
  echo "WAN_WORKER_API_KEY=${KEY}" >> "${ROOT}/.env"
fi
# Idempotent env keys (do not append duplicates on re-run).
upsert_env() {
  local key="$1"
  local value="$2"
  if grep -q "^${key}=" "${ROOT}/.env" 2>/dev/null; then
    # shellcheck disable=SC2016
    sed -i "s|^${key}=.*|${key}=${value}|" "${ROOT}/.env"
  else
    echo "${key}=${value}" >> "${ROOT}/.env"
  fi
}
upsert_env "LTX_PYTHON" "ltx_service/engines/ltx-env/bin/python"
upsert_env "LTX_ENV_DIR" "ltx_service/engines/ltx-env"
upsert_env "LTX_REPO_DIR" "ltx_service/third_party/LTX-Video"
upsert_env "LTX_MODEL_PATH" "ltx_service/models/ltx/ltxv-2b-0.9.6-distilled-04-25.safetensors"
upsert_env "LTX_PIPELINE_CONFIG" "configs/ltxv-2b-0.9.6-distilled.yaml"
upsert_env "WAN_PERSISTENT_INFERENCE" "false"
upsert_env "HF_HUB_ENABLE_HF_TRANSFER" "0"

echo "LTX setup complete. Test: python scripts/test_ltx.py --prompt '...' --duration 5"
echo "Health: GET /api/v1/video/engines/ltx/health"
