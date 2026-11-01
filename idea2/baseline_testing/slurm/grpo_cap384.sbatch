#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --job-name=social-9b-grpo-cap384
#SBATCH --output=logs/grpo-cap384-%j.log
#SBATCH --error=logs/grpo-cap384-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
ADAPTER_PATH=$(python -c 'import json; from social_rl.common import STORE; print(json.load(open(STORE/"runs/grpo/latest.json"))["checkpoint"])')
python scripts/evaluate.py --manifest "$SOCIAL_RL_STORE/data/manifests/ego_midterm_panel_accessible.jsonl" \
  --output "$SOCIAL_RL_STORE/runs/grpo_ego_cap384_video400" --adapter "$ADAPTER_PATH" \
  --conditions video --max-new-tokens 384 --max-hours 3.4
N=$(wc -l < "$SOCIAL_RL_STORE/runs/grpo_ego_cap384_video400/predictions.jsonl")
if [[ "$N" -ne 400 ]]; then
  echo "Incomplete GRPO 384-token EgoNormia run: $N/400" >&2
  exit 1
fi
