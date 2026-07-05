#!/usr/bin/env bash
# Sync this project to the remote RTX 3090 Ti box and run a command under its GPU venv.
#
# The heavy RAPIDS/GPU pipeline (cuDF target encoding, GPU GBDTs, NN zoo) runs on the box;
# the Mac has no CUDA. Code is pushed with rsync (the box is not a git checkout); the
# git-ignored data/ (raw + original) and generated experiments/artifacts/ are synced too so
# training and blending have their inputs and we can pull results back.
#
# Usage:
#   scripts/remote_run.sh push                       # sync code + data up to the box
#   scripts/remote_run.sh run  <cmd...>              # run <cmd> in the box venv (cwd = project)
#   scripts/remote_run.sh pull                       # sync experiments/artifacts + outputs back
#   scripts/remote_run.sh all  <cmd...>              # push, run, then pull
#
# Example:
#   scripts/remote_run.sh all python -m src.train --model xgb --backend gpu --te-order 4 \
#       --sample 100000 --tag te2
set -euo pipefail

BOX="${SHR_BOX:-kamron@192.168.0.92}"
REMOTE_DIR="${SHR_REMOTE_DIR:-StudentHealthRisk}"     # relative to the box user's ~
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The box authenticates with the passphrase-less `dragon` key; use it directly so this works in a
# fresh shell (no ssh-agent needed). Override with SHR_SSH_KEY.
SSH_KEY="${SHR_SSH_KEY:-$HOME/.ssh/dragon}"
SSH="ssh -i $SSH_KEY -o ConnectTimeout=10"

# Files/dirs never worth shipping. Code + data/{raw,original} + artifacts DO sync.
RSYNC_EXCLUDES=(--exclude '.venv' --exclude '.git' --exclude '__pycache__'
                --exclude '*.pyc' --exclude '.pytest_cache' --exclude 'catboost_info'
                --exclude 'data/processed' --exclude 'reports/figures' --exclude 'models')

push() {
  echo ">> push code+data -> $BOX:~/$REMOTE_DIR"
  rsync -az -e "$SSH" --delete "${RSYNC_EXCLUDES[@]}" \
    --exclude 'experiments/artifacts' \
    "$LOCAL_DIR/" "$BOX:$REMOTE_DIR/"
  # artifacts synced without --delete so box-side results are never clobbered by the Mac
  rsync -az -e "$SSH" "$LOCAL_DIR/experiments/artifacts/" "$BOX:$REMOTE_DIR/experiments/artifacts/" 2>/dev/null || true
}

run() {
  echo ">> run on $BOX: $*"
  $SSH "$BOX" "bash -lc 'cd $REMOTE_DIR && source .venv/bin/activate && $*'"
}

pull() {
  echo ">> pull artifacts+outputs <- $BOX"
  mkdir -p "$LOCAL_DIR/experiments/artifacts" "$LOCAL_DIR/outputs"
  rsync -az -e "$SSH" "$BOX:$REMOTE_DIR/experiments/artifacts/" "$LOCAL_DIR/experiments/artifacts/"
  rsync -az -e "$SSH" "$BOX:$REMOTE_DIR/outputs/" "$LOCAL_DIR/outputs/" 2>/dev/null || true
}

cmd="${1:-}"; shift || true
case "$cmd" in
  push) push ;;
  run)  run "$@" ;;
  pull) pull ;;
  all)  push; run "$@"; pull ;;
  *) echo "usage: $0 {push|run|pull|all} [cmd...]" >&2; exit 2 ;;
esac
