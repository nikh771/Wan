#!/usr/bin/env bash
# Explicit download of zai-org/CogVideoX-2b into cogvideox_service/models/cogvideox/CogVideoX-2b
# Never called from FastAPI startup.
set -euo pipefail

unset HF_HUB_ENABLE_HF_TRANSFER
export HF_HUB_ENABLE_HF_TRANSFER=0

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

MODEL_ID="${COGVIDEOX_HF_ID:-zai-org/CogVideoX-2b}"
DEST="${COGVIDEOX_MODEL_PATH:-cogvideox_service/models/cogvideox/CogVideoX-2b}"
if [[ "${DEST}" != /* ]]; then
  DEST="${ROOT}/${DEST}"
fi

mkdir -p "${DEST}"

PYTHON_BIN="${COGVIDEOX_PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${ROOT}/cogvideox_service/engines/cogvideox-env/bin/python" ]]; then
    PYTHON_BIN="${ROOT}/cogvideox_service/engines/cogvideox-env/bin/python"
  elif [[ -x "${ROOT}/ltx_service/engines/ltx-env/bin/python" ]]; then
    PYTHON_BIN="${ROOT}/ltx_service/engines/ltx-env/bin/python"
  elif [[ -x "${ROOT}/venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT}/venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
elif [[ "${PYTHON_BIN}" != /* ]]; then
  PYTHON_BIN="${ROOT}/${PYTHON_BIN}"
fi

echo "==> Downloading ${MODEL_ID} → ${DEST}"
echo "    Using Python: ${PYTHON_BIN}"

export MODEL_ID DEST
"${PYTHON_BIN}" -c "
from huggingface_hub import snapshot_download
import os
path = snapshot_download(
    repo_id=os.environ['MODEL_ID'],
    local_dir=os.environ['DEST'],
    local_dir_use_symlinks=False,
)
print('Downloaded to:', path)
"

test -f "${DEST}/model_index.json"
echo "OK: ${DEST}/model_index.json present"
