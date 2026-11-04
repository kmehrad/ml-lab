#!/usr/bin/env bash
# Optional GPU host workflow. No deletion-based synchronization.
set -euo pipefail

BOX="${EVP_BOX:-kamron@192.168.0.92}"
REMOTE_DIR="${EVP_REMOTE_DIR:-ElectricVehiclePurchases}"
SSH_KEY="${EVP_SSH_KEY:-$HOME/.ssh/dragon}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ssh_cmd=(ssh -i "$SSH_KEY" -o ConnectTimeout=10)
rsync_ssh="ssh -i $SSH_KEY -o ConnectTimeout=10"
excludes=(--exclude .venv --exclude .git --exclude __pycache__ --exclude .pytest_cache
          --exclude catboost_info --exclude models --exclude reports/figures)

push() {
  rsync -az -e "$rsync_ssh" "${excludes[@]}" "$LOCAL_DIR/" "$BOX:$REMOTE_DIR/"
}

run() {
  "${ssh_cmd[@]}" "$BOX" "cd '$REMOTE_DIR' && uv sync && uv run $*"
}

pull() {
  mkdir -p "$LOCAL_DIR/experiments/artifacts" "$LOCAL_DIR/outputs"
  rsync -az -e "$rsync_ssh" "$BOX:$REMOTE_DIR/experiments/artifacts/" "$LOCAL_DIR/experiments/artifacts/"
  rsync -az -e "$rsync_ssh" "$BOX:$REMOTE_DIR/outputs/" "$LOCAL_DIR/outputs/"
}

case "${1:-}" in
  push) push ;;
  run) shift; run "$@" ;;
  pull) pull ;;
  all) shift; push; run "$@"; pull ;;
  *) echo "usage: $0 {push|run|pull|all} [uv-run-command...]" >&2; exit 2 ;;
esac
