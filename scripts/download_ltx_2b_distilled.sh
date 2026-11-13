#!/usr/bin/env bash
# Download LTX 2B distilled 0.9.6 checkpoint into ltx_service/models/ltx/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/ltx_service/models/ltx"
mkdir -p "${OUT}"

# Avoid empty/partial HF downloads when hf_transfer is enabled but not installed.
unset HF_HUB_ENABLE_HF_TRANSFER
export HF_HUB_ENABLE_HF_TRANSFER=0

FILE="ltxv-2b-0.9.6-distilled-04-25.safetensors"

if command -v hf >/dev/null 2>&1; then
  hf download Lightricks/LTX-Video "${FILE}" --local-dir "${OUT}"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download Lightricks/LTX-Video "${FILE}" --local-dir "${OUT}"
else
  echo "Install huggingface_hub CLI: pip install huggingface_hub"
  exit 1
fi

echo "Saved to ${OUT}/${FILE}"
echo "Set LTX_MODEL_PATH=ltx_service/models/ltx/${FILE} in .env"
