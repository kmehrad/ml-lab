#!/usr/bin/env bash
# Launch a detached DeBERTa fold training run on the workstation and return.
# Usage: bash run_fold.sh <fold|all> <epochs> <logfile>
set -euo pipefail
cd "$HOME/LLM-clf-FT"
FOLD="${1:-0}"
EPOCHS="${2:-2}"
LOG="${3:-logs/deberta_fold${FOLD}.log}"
mkdir -p logs
pkill -9 -f "m src.train_deberta" 2>/dev/null || true
sleep 2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
PY="$HOME/anaconda3/envs/llmclf/bin/python"
setsid "$PY" -u -m src.train_deberta --fold "$FOLD" --epochs "$EPOCHS" \
    > "$LOG" 2>&1 < /dev/null &
echo "launched fold=$FOLD epochs=$EPOCHS pid=$! log=$LOG"
