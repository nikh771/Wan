#!/usr/bin/env bash
# Download Tencent HunyuanVideo-1.5 checkpoints + documented text encoders.
set -euo pipefail

unset HF_HUB_ENABLE_HF_TRANSFER
export HF_HUB_ENABLE_HF_TRANSFER=0

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DEST="${HUNYUAN_MODEL_PATH:-hunyuan_service/models/hunyuan/HunyuanVideo-1.5}"
if [[ "${DEST}" != /* ]]; then
  DEST="${ROOT}/${DEST}"
fi
mkdir -p "${DEST}"

PYTHON_BIN="${HUNYUAN_PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${ROOT}/hunyuan_service/engines/hunyuan-env/bin/python" ]]; then
    PYTHON_BIN="${ROOT}/hunyuan_service/engines/hunyuan-env/bin/python"
  elif [[ -x "${ROOT}/venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT}/venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
elif [[ "${PYTHON_BIN}" != /* ]]; then
  PYTHON_BIN="${ROOT}/${PYTHON_BIN}"
fi

echo "==> Downloading tencent/HunyuanVideo-1.5 → ${DEST}"
"${PYTHON_BIN}" -m pip install -q -U "huggingface_hub[cli]" 2>/dev/null || true

hf_download() {
  if command -v hf >/dev/null 2>&1; then
    hf download "$@"
  else
    "${PYTHON_BIN}" -m huggingface_hub.cli.hf download "$@"
  fi
}

hf_download tencent/HunyuanVideo-1.5 --local-dir "${DEST}"

ENC_BASE="${DEST}/text_encoder"
mkdir -p "${ENC_BASE}"

echo "==> Text encoder: Qwen2.5-VL-7B-Instruct"
hf_download Qwen/Qwen2.5-VL-7B-Instruct --local-dir "${ENC_BASE}/llm"

echo "==> Text encoder: byt5-small"
hf_download google/byt5-small --local-dir "${ENC_BASE}/byt5-small"

if command -v modelscope >/dev/null 2>&1; then
  echo "==> Text encoder: Glyph-SDXL-v2 (ModelScope)"
  modelscope download --model AI-ModelScope/Glyph-SDXL-v2 \
    --local_dir "${ENC_BASE}/Glyph-SDXL-v2"
else
  echo "WARN: modelscope CLI not found — install in hunyuan-env for Glyph-SDXL-v2"
  echo "      modelscope download --model AI-ModelScope/Glyph-SDXL-v2 --local_dir ${ENC_BASE}/Glyph-SDXL-v2"
fi

test -d "${DEST}/transformer" || test -n "$(ls -A "${DEST}" 2>/dev/null)"
echo "OK: Hunyuan checkpoints under ${DEST}"