#!/bin/bash
export SOCIAL_RL_STORE=${SOCIAL_RL_STORE:-/ocean/projects/cis260338p/apatawar/social_norm_rl}
export HF_HOME="$SOCIAL_RL_STORE/cache/huggingface"
export PIP_CACHE_DIR="$SOCIAL_RL_STORE/cache/pip"
SOCIAL_RL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$SOCIAL_RL_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export LD_LIBRARY_PATH=/opt/packages/anaconda3-2024.10-1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$SOCIAL_RL_STORE/envs/qwen35/bin/activate"
cd "$SOCIAL_RL_ROOT"
