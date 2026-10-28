#!/usr/bin/env bash
# Download Wan 2.2 TI2V-5B weights only (does not run inference).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${WAN_CKPT_DIR:-${ROOT}/Wan2.2-TI2V-5B}"

echo "Target checkpoint directory: ${CKPT_DIR}"
mkdir -p "${CKPT_DIR}"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "Install the Hugging Face CLI first: pip install 'huggingface_hub[cli]'"
  exit 1
fi

huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir "${CKPT_DIR}"
echo "Done. Set WAN_CKPT_DIR=${CKPT_DIR} in .env on your linux_cuda host."
