#!/usr/bin/env bash
# Storyboard engine setup (CPU, no diffusion weights). Run from repo root:
#   bash scripts/runpod_remote_setup_storyboard.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

echo "==> Storyboard engine setup | project root: ${ROOT}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y --no-install-recommends ffmpeg
  else
    echo "ERROR: ffmpeg required."
    exit 1
  fi
fi

python3 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-storyboard.txt

if [[ ! -f "${ROOT}/.env" ]]; then
  cp .env.example .env
  KEY="$(python -c "import secrets; print(secrets.token_urlsafe(32))")"
  {
    echo ""
    echo "WAN_WORKER_API_KEY=${KEY}"
    echo "WAN_PERSISTENT_INFERENCE=false"
    echo "WAN_JOBS_DIR=${ROOT}/wan_jobs"
  } >> .env
  chmod 600 .env
fi

echo "Storyboard setup complete. Create jobs with engine_type=storyboard"
