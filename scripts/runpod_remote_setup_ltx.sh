#!/usr/bin/env bash
# LTX-Video 2B distilled — full setup in one run (prefer: bash scripts/runpod_remote_setup.sh ltx).
set -euo pipefail

unset HF_HUB_ENABLE_HF_TRANSFER
export HF_HUB_ENABLE_HF_TRANSFER=0

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export ROOT

if [[ ! -f "${ROOT}/scripts/gpu_setup_common.sh" ]]; then
  echo "ERROR: Missing scripts/gpu_setup_common.sh — re-clone the repo if /workspace/Wan was deleted."
  exit 1
fi

# shellcheck source=gpu_setup_common.sh
source "${ROOT}/scripts/gpu_setup_common.sh"

export GPU_SETUP_TARGET_ENGINE="${GPU_SETUP_TARGET_ENGINE:-ltx}"
export GPU_SETUP_FULL_SWITCH="${GPU_SETUP_FULL_SWITCH:-1}"
bash "${ROOT}/scripts/gpu_setup_clean_disk.sh"
gpu_require_nvidia
gpu_ensure_ffmpeg
gpu_ensure_git

LTX_DIR="${ROOT}/ltx_service/third_party/LTX-Video"
LTX_VENV="${ROOT}/ltx_service/engines/ltx-env"
MODEL_DIR="${ROOT}/ltx_service/models/ltx"
mkdir -p "${MODEL_DIR}" "${ROOT}/ltx_service/engines"

echo "==> LTX 2B distilled setup | ${ROOT}"
echo "==> Free disk: $(gpu_free_gb "${ROOT}") GB"

echo "==> API venv (requirements.txt)..."
gpu_recreate_venv "${ROOT}/venv"
gpu_install_requirements "${ROOT}/venv" "${ROOT}/requirements.txt"

echo "==> LTX worker venv..."
gpu_recreate_venv "${LTX_VENV}"
# shellcheck disable=SC1091
source "${LTX_VENV}/bin/activate"
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

if [[ ! -d "${LTX_DIR}/.git" ]]; then
  echo "==> Cloning LTX-Video..."
  git clone --depth 1 https://github.com/Lightricks/LTX-Video.git "${LTX_DIR}"
fi
python -m pip install -e "${LTX_DIR}[inference]"

echo "==> Downloading LTX distilled checkpoint..."
bash "${ROOT}/scripts/download_ltx_2b_distilled.sh"

echo "==> Prefetching PixArt T5 text_encoder + tokenizer..."
LTX_PYTHON="${LTX_VENV}/bin/python" bash "${ROOT}/scripts/download_ltx_pixart_text_encoder.sh"

gpu_ensure_api_key
gpu_upsert_env "LTX_PYTHON" "ltx_service/engines/ltx-env/bin/python"
gpu_upsert_env "LTX_ENV_DIR" "ltx_service/engines/ltx-env"
gpu_upsert_env "LTX_REPO_DIR" "ltx_service/third_party/LTX-Video"
gpu_upsert_env "LTX_MODEL_PATH" "ltx_service/models/ltx/ltxv-2b-0.9.6-distilled-04-25.safetensors"
gpu_upsert_env "LTX_PIPELINE_CONFIG" "configs/ltxv-2b-0.9.6-distilled.yaml"
gpu_upsert_env "WAN_PERSISTENT_INFERENCE" "false"
gpu_upsert_env "HF_HUB_ENABLE_HF_TRANSFER" "0"

gpu_print_start_api
echo "    Test:  python scripts/test_ltx.py --prompt '...' --duration 5"
echo "    Jobs:  engine_type=ltx on POST /video/generate"