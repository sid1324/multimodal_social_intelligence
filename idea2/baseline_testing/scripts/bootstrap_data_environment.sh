#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export SOCIAL_RL_STORE=${SOCIAL_RL_STORE:-/ocean/projects/cis260338p/apatawar/social_norm_rl}
export PIP_CACHE_DIR="$SOCIAL_RL_STORE/cache/pip"
export LD_LIBRARY_PATH=/opt/packages/anaconda3-2024.10-1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
PY="$SOCIAL_RL_STORE/envs/data/bin/python"
if [[ ! -x "$PY" ]]; then
  mkdir -p "$SOCIAL_RL_STORE/envs"
  /opt/packages/anaconda3-2024.10-1/bin/python -m venv --system-site-packages "$SOCIAL_RL_STORE/envs/data"
fi
"$PY" -m pip install 'yt-dlp==2026.10.21' 'hf_xet>=1.1'
"$PY" -c 'import av, pyarrow, pandas, requests, huggingface_hub, yt_dlp; print("DATA_ENV_READY")'
touch "$SOCIAL_RL_STORE/envs/data/READY"
