#!/usr/bin/env bash
# HunyuanVideo-1.5 — full setup (prefer: bash scripts/runpod_remote_setup.sh hunyuan).
set -euo pipefail

unset HF_HUB_ENABLE_HF_TRANSFER
export HF_HUB_ENABLE_HF_TRANSFER=0

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export ROOT

# shellcheck source=gpu_setup_common.sh
source "${ROOT}/scripts/gpu_setup_common.sh"

export GPU_SETUP_TARGET_ENGINE="${GPU_SETUP_TARGET_ENGINE:-hunyuan}"
export GPU_SETUP_FULL_SWITCH="${GPU_SETUP_FULL_SWITCH:-1}"
bash "${ROOT}/scripts/gpu_setup_clean_disk.sh"
gpu_require_nvidia
gpu_ensure_git

REPO_DIR="${ROOT}/hunyuan_service/third_party/HunyuanVideo-1.5"
HY_VENV="${ROOT}/hunyuan_service/engines/hunyuan-env"
mkdir -p "${ROOT}/hunyuan_service/models/hunyuan" "${ROOT}/hunyuan_service/engines" "${ROOT}/hunyuan_service/outputs"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  echo "==> Cloning HunyuanVideo-1.5..."
  mkdir -p "$(dirname "${REPO_DIR}")"
  git clone --depth 1 https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5.git "${REPO_DIR}"
fi

echo "==> API venv (requirements.txt)..."
gpu_recreate_venv "${ROOT}/venv"
gpu_install_requirements "${ROOT}/venv" "${ROOT}/requirements.txt"

echo "==> Hunyuan worker venv..."
gpu_recreate_venv "${HY_VENV}"
# shellcheck disable=SC1091
source "${HY_VENV}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "${REPO_DIR}/requirements.txt"
python -m pip install -i https://mirrors.tencent.com/pypi/simple/ --upgrade tencentcloud-sdk-python || true
python -m pip install -r "${ROOT}/hunyuan_service/requirements-gpu.txt"
python -m pip install -U "huggingface_hub[cli]" modelscope || true

gpu_ensure_api_key
gpu_upsert_env "HUNYUAN_PYTHON" "hunyuan_service/engines/hunyuan-env/bin/python"
gpu_upsert_env "HUNYUAN_ENV_DIR" "hunyuan_service/engines/hunyuan-env"
gpu_upsert_env "HUNYUAN_REPO_DIR" "hunyuan_service/third_party/HunyuanVideo-1.5"
gpu_upsert_env "HUNYUAN_MODEL_PATH" "hunyuan_service/models/hunyuan/HunyuanVideo-1.5"
gpu_upsert_env "HUNYUAN_RESOLUTION" "480p"
gpu_upsert_env "HUNYUAN_ENABLE_OFFLOAD" "true"
gpu_upsert_env "HUNYUAN_PROMPT_REWRITE" "false"
gpu_upsert_env "HUNYUAN_ENABLE_SR" "false"
gpu_upsert_env "HUNYUAN_ENABLE_STEP_DISTILL" "true"
gpu_upsert_env "HUNYUAN_STEP_DISTILL_STEPS" "8"
gpu_upsert_env "WAN_PERSISTENT_INFERENCE" "false"
gpu_upsert_env "HF_HUB_ENABLE_HF_TRANSFER" "0"

echo "==> Downloading Hunyuan weights (large; may take a while)..."
HUNYUAN_PYTHON="${ROOT}/hunyuan_service/engines/hunyuan-env/bin/python" \
  bash "${ROOT}/scripts/download_hunyuan_1_5.sh"

gpu_print_start_api
echo "    Health: curl -s 'http://127.0.0.1:8080/health?engine_type=hunyuan'"
echo "    Jobs:   engine_type=hunyuan on POST /video/generate"