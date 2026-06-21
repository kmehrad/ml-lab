#!/usr/bin/env bash
# Build an isolated conda env with a known-good DeBERTa training stack.
# transformers 5.10 on this box has broken DeBERTa-v2 gradients; 4.46 is stable.
set -euo pipefail
CONDA="$HOME/anaconda3/bin/conda"
"$CONDA" create -y -n llmclf python=3.11
PIP="$HOME/anaconda3/envs/llmclf/bin/pip"
"$PIP" install --upgrade pip
"$PIP" install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
"$PIP" install "transformers==4.46.3" "tokenizers<0.21" \
    datasets accelerate sentencepiece protobuf \
    scikit-learn pandas pyarrow tiktoken
echo "ENV_BUILD_DONE"
"$HOME/anaconda3/envs/llmclf/bin/python" -c "import torch,transformers; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'tf',transformers.__version__)"
