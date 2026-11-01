#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --job-name=social-muson-rl-smoke
#SBATCH --output=logs/muson-train-pilot-%j.log
#SBATCH --error=logs/muson-train-pilot-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
ARGS=()
if test -f "$SOCIAL_RL_STORE/runs/rloo_muson_pilot/latest.json"; then ARGS+=(--resume); fi
python scripts/train_muson_rloo.py \
  --manifest "$SOCIAL_RL_STORE/data/muson/train.jsonl" \
  --output "$SOCIAL_RL_STORE/runs/rloo_muson_pilot" \
  --start-adapter "$SOCIAL_RL_STORE/runs/rloo/checkpoint-00100" \
  --steps 2 --max-hours 0.7 "${ARGS[@]}"
python - <<'PY'
import json
from social_rl.common import STORE
status=json.load(open(STORE/'runs/rloo_muson_pilot/status.json'))
assert status['complete_requested_steps'], status
PY
