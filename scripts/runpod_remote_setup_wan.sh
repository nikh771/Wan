#!/usr/bin/env bash
# Wan 2.2 TI2V-5B — full setup in one run (prefer: bash scripts/runpod_remote_setup.sh wan).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export ROOT
export GPU_SETUP_TARGET_ENGINE="${GPU_SETUP_TARGET_ENGINE:-wan}"
export GPU_SETUP_FULL_SWITCH="${GPU_SETUP_FULL_SWITCH:-1}"
bash "${ROOT}/scripts/gpu_setup_clean_disk.sh"

# shellcheck source=gpu_setup_common.sh
source "${ROOT}/scripts/gpu_setup_common.sh"
gpu_require_nvidia
gpu_ensure_ffmpeg

CKPT_DIR="${ROOT}/wan_service/models/Wan2.2-TI2V-5B"
JOBS_DIR="${ROOT}/wan_jobs"
if [[ -d /workspace ]]; then
  WS_AVAIL_GB="$(df -BG /workspace 2>/dev/null | awk 'NR==2 {gsub(/G/,"",$4); print $4}' || echo 0)"
  if [[ "${WS_AVAIL_GB}" -ge 35 ]]; then
    CKPT_DIR="/workspace/Wan/wan_service/models/Wan2.2-TI2V-5B"
    JOBS_DIR="/workspace/wan_jobs"
    echo "==> Using /workspace for weights (${WS_AVAIL_GB} GB free)"
  fi
fi
mkdir -p "${JOBS_DIR}"

echo "==> Wan engine setup | ${ROOT}"
echo "==> Free disk: $(gpu_free_gb "${ROOT}") GB"

echo "==> API + inference venv (requirements-gpu.txt + flash-attn)..."
gpu_recreate_venv "${ROOT}/venv"
gpu_install_requirements "${ROOT}/venv" "${ROOT}/requirements-gpu.txt"
# shellcheck disable=SC1091
source "${ROOT}/venv/bin/activate"
python -m pip install --no-deps --only-binary=flash-attn -r requirements-gpu-flash.txt

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda)"
python -c "import flash_attn; print('flash_attn:', flash_attn.__version__)"

if [[ ! -f wan_service/third_party/Wan2.2/generate.py ]]; then
  echo "ERROR: wan_service/third_party/Wan2.2/generate.py missing (vendor Wan in repo)."
  exit 1
fi

gpu_ensure_api_key
gpu_upsert_env "WAN_PYTHON" "venv/bin/python"
gpu_upsert_env "WAN_CKPT_DIR" "${CKPT_DIR}"
gpu_upsert_env "WAN_JOBS_DIR" "${JOBS_DIR}"
gpu_upsert_env "WAN_PERSISTENT_INFERENCE" "false"
gpu_upsert_env "HF_HUB_ENABLE_HF_TRANSFER" "0"

export WAN_CKPT_DIR="${CKPT_DIR}"
echo "==> Downloading Wan TI2V-5B weights → ${CKPT_DIR}"
bash scripts/download_ti2v_5b_weights.sh

gpu_print_start_api
echo "    engine_type=wan on POST /video/generate"