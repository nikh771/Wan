#!/usr/bin/env bash
# Install/start Ollama and pull the small model used for LTX prompt enhancement.
#   bash scripts/setup_ollama_ltx_prompts.sh
set -euo pipefail

MODEL="${LTX_OLLAMA_MODEL:-llama3.2:3b}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "==> Installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
fi

if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "==> Starting ollama serve in background..."
  nohup ollama serve >/tmp/ollama-serve.log 2>&1 &
  for _ in $(seq 1 30); do
    if curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
fi

if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "ERROR: Ollama API not reachable at http://127.0.0.1:11434"
  echo "Check /tmp/ollama-serve.log"
  exit 1
fi

echo "==> Pulling ${MODEL} (used with num_gpu=0 + keep_alive=0 so LTX keeps the GPU)..."
ollama pull "${MODEL}"

echo "Ollama ready for LTX_PROMPT_ENHANCE. Model=${MODEL}"
echo "Test: curl -s http://127.0.0.1:11434/api/tags | python3 -m json.tool"
