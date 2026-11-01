#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --job-name=social-normlens-rloo
#SBATCH --output=logs/normlens-train-%j.log
#SBATCH --error=logs/normlens-train-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
ARGS=()
if test -f "$SOCIAL_RL_STORE/runs/rloo_normlens/latest.json"; then ARGS+=(--resume); fi
python scripts/train_norm_rloo.py \
  --manifest "$SOCIAL_RL_STORE/data/normlens/train.jsonl" \
  --output "$SOCIAL_RL_STORE/runs/rloo_normlens" \
  --start-adapter "$SOCIAL_RL_STORE/runs/rloo/checkpoint-00100" \
  --steps 25 --max-hours 3.4 "${ARGS[@]}"
python - <<'PY'
import json
from social_rl.common import STORE
status=json.load(open(STORE/'runs/rloo_normlens/status.json'))
assert status['complete_requested_steps'], status
PY
