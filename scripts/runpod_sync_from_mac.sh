#!/usr/bin/env bash
# Push local repo to RunPod over SSH (excludes venv, weights, job artifacts).
# Usage: bash scripts/runpod_sync_from_mac.sh [ssh-host-alias]
# Default host alias: runpod-wan (see ~/.ssh/config)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${1:-runpod-wan}"
REMOTE_DIR="${RUNPOD_REMOTE_DIR:-/workspace/Wan}"

echo "Syncing ${ROOT} -> ${HOST}:${REMOTE_DIR}"

rsync -avz --delete \
  --exclude '.git/' \
  --exclude 'venv/' \
  --exclude '.venv/' \
  --exclude 'models/' \
  --exclude 'wan_jobs/' \
  --exclude '.env' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  -e "ssh -i ${HOME}/.ssh/id_ed25519" \
  "${ROOT}/" "${HOST}:${REMOTE_DIR}/"

echo "Done. SSH in and run: cd ${REMOTE_DIR} && bash scripts/runpod_remote_setup.sh"
