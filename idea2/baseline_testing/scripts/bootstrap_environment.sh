#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export SOCIAL_RL_STORE=${SOCIAL_RL_STORE:-/ocean/projects/cis260338p/apatawar/social_norm_rl}
export PIP_CACHE_DIR="$SOCIAL_RL_STORE/cache/pip"
export LD_LIBRARY_PATH=/opt/packages/anaconda3-2024.10-1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
PY="$SOCIAL_RL_STORE/envs/qwen35/bin/python"
if [[ ! -x "$PY" ]]; then
  mkdir -p "$SOCIAL_RL_STORE/envs"
  /opt/packages/anaconda3-2024.10-1/bin/python -m venv "$SOCIAL_RL_STORE/envs/qwen35"
fi
# Reuse complete wheels salvaged from this task's interrupted installation.
if compgen -G "$SOCIAL_RL_STORE/cache/wheelhouse/*.whl" > /dev/null; then
  "$PY" -m pip install --no-deps "$SOCIAL_RL_STORE"/cache/wheelhouse/*.whl
fi
"$PY" -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
"$PY" -m pip install -r requirements.txt
"$PY" -m pip install -e .
"$PY" -m pip freeze > reports/environment-lock.txt
"$PY" -c 'import torch, transformers, peft, bitsandbytes; assert "sm_70" in torch._C._cuda_getArchFlags(); print(torch.__version__, transformers.__version__)'
touch "$SOCIAL_RL_STORE/envs/qwen35/READY"
