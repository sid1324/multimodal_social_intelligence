#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --job-name=social-9b-grpo-main
#SBATCH --output=logs/grpo-main-%j.log
#SBATCH --error=logs/grpo-main-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
python - <<'PY'
import json
from social_rl.common import STORE
status = json.loads((STORE / 'runs/grpo_pilot/status.json').read_text())
if status['optimized_groups'] != 2 or not status['complete_requested_groups']:
    raise ValueError(f'GRPO pilot did not complete: {status}')
PY
python scripts/train_grpo.py \
  --manifest "$SOCIAL_RL_STORE/data/manifests/r3_train_accessible.jsonl" \
  --output "$SOCIAL_RL_STORE/runs/grpo" --groups 100 --epochs 2 \
  --clip-epsilon 0.2 --max-hours 3.4
python - <<'PY'
import json
from social_rl.common import STORE
status = json.loads((STORE / 'runs/grpo/status.json').read_text())
if status['optimized_groups'] != 100 or not status['complete_requested_groups']:
    raise ValueError(f'GRPO training did not reach 100 groups: {status}')
PY
