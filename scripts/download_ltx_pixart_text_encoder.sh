#!/usr/bin/env bash
# Prefetch PixArt T5 text_encoder + tokenizer used by LTX inference
# (PixArt-alpha/PixArt-XL-2-1024-MS). Safe to re-run; skips completed files.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ID="PixArt-alpha/PixArt-XL-2-1024-MS"

# Broken/partial downloads happen when this is set but hf_transfer is missing.
unset HF_HUB_ENABLE_HF_TRANSFER
export HF_HUB_ENABLE_HF_TRANSFER=0

# Prefer the isolated LTX env when present (has huggingface_hub / transformers).
PYTHON_BIN="${LTX_PYTHON:-}"
if [[ -z "${PYTHON_BIN}" && -x "${ROOT}/ltx_service/engines/ltx-env/bin/python" ]]; then
  PYTHON_BIN="${ROOT}/ltx_service/engines/ltx-env/bin/python"
elif [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

echo "==> Prefetching ${REPO_ID} text_encoder + tokenizer via ${PYTHON_BIN}"
echo "    HF_HOME=${HF_HOME:-"(default ~/.cache/huggingface)"}"

"${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

repo_id = "PixArt-alpha/PixArt-XL-2-1024-MS"
path = Path(
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=["text_encoder/*", "tokenizer/*"],
    )
)
te = path / "text_encoder"
tok = path / "tokenizer"
shards = sorted(te.glob("model-*-of-*.safetensors"))
index = te / "model.safetensors.index.json"
missing = []
if not te.is_dir():
    missing.append(str(te))
if not tok.is_dir():
    missing.append(str(tok))
if not index.is_file():
    missing.append(str(index))
if len(shards) < 2:
    missing.append(f"{te}/model-*-of-*.safetensors (need >=2 shards, found {len(shards)})")
for shard in shards:
    # Incomplete hub downloads leave tiny/zero-byte blobs; real shards are multi-GB.
    if shard.stat().st_size < 1_000_000_000:
        missing.append(f"{shard} too small ({shard.stat().st_size} bytes)")
if missing:
    print("ERROR: PixArt text encoder download incomplete:", file=sys.stderr)
    for item in missing:
        print(f"  - {item}", file=sys.stderr)
    print(
        "Remove the incomplete cache under $HF_HOME/hub/models--PixArt-alpha--PixArt-XL-2-1024-MS and re-run.",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"OK: {path}")
for shard in shards:
    print(f"  shard {shard.name}: {shard.stat().st_size / (1024**3):.2f} GiB")
PY

echo "PixArt text encoder ready (used automatically by LTX on first job)."
