#!/usr/bin/env bash
set -euo pipefail

BOX="${SMARTPHONE_BOX:-pyml}"
REMOTE_ROOT="${SMARTPHONE_REMOTE_ROOT:-SmartphoneAddiction}"
LOCAL_MEMBER="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXCLUDES=(--exclude '.venv' --exclude '.git' --exclude '__pycache__' --exclude '*.pyc'
          --exclude '.pytest_cache' --exclude 'catboost_info' --exclude 'data/processed'
          --exclude 'reports/figures' --exclude 'models' --exclude 'outputs')

push() {
  ssh "$BOX" "mkdir -p '$REMOTE_ROOT'"
  rsync -az --delete "${EXCLUDES[@]}" --exclude '.conda-env' --exclude 'experiments/artifacts' "$LOCAL_MEMBER/" "$BOX:$REMOTE_ROOT/"
  rsync -az "$LOCAL_MEMBER/experiments/artifacts/" "$BOX:$REMOTE_ROOT/experiments/artifacts/" 2>/dev/null || true
}
setup() {
  ssh "$BOX" "bash -lc '
    test -x $REMOTE_ROOT/.conda-env/bin/python || /home/pyml/anaconda3/bin/conda create -y -p $REMOTE_ROOT/.conda-env python=3.11 pip
    $REMOTE_ROOT/.conda-env/bin/pip install numpy pandas scipy scikit-learn pyarrow xgboost catboost lightgbm optuna
  '"
}
run() {
  local command="$*"
  ssh "$BOX" "bash -lc 'cd $REMOTE_ROOT && export PATH=\$PWD/.conda-env/bin:\$PATH && $command'"
}
pull() {
  mkdir -p "$LOCAL_MEMBER/experiments/artifacts" "$LOCAL_MEMBER/outputs"
  rsync -az "$BOX:$REMOTE_ROOT/experiments/artifacts/" "$LOCAL_MEMBER/experiments/artifacts/"
  rsync -az "$BOX:$REMOTE_ROOT/outputs/" "$LOCAL_MEMBER/outputs/" 2>/dev/null || true
}
command="${1:-}"; shift || true
case "$command" in
  push) push;; setup) setup;; run) run "$@";; pull) pull;; all) push; setup; run "$@"; pull;;
  *) echo "usage: $0 {push|setup|run|pull|all} [command...]" >&2; exit 2;;
esac
