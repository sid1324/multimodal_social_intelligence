#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=04:30:00
#SBATCH --job-name=social-muson-eval
#SBATCH --output=logs/muson-eval-%j.log
#SBATCH --error=logs/muson-eval-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
ADAPTER="$SOCIAL_RL_STORE/runs/rloo_muson/checkpoint-00050"
python scripts/evaluate_muson.py \
  --manifest "$SOCIAL_RL_STORE/data/muson/val.jsonl" \
  --output "$SOCIAL_RL_STORE/runs/rloo_muson_val300" \
  --adapter "$ADAPTER" --limit 300 --max-hours 1.4
N=$(wc -l < "$SOCIAL_RL_STORE/runs/rloo_muson_val300/predictions.jsonl")
test "$N" -eq 300
python scripts/evaluate.py \
  --manifest "$SOCIAL_RL_STORE/data/manifests/ego_midterm_panel_accessible.jsonl" \
  --output "$SOCIAL_RL_STORE/runs/rloo_muson_ego_cap1024_video400" \
  --adapter "$ADAPTER" --conditions video --max-new-tokens 1024 --max-hours 3.2
N=$(wc -l < "$SOCIAL_RL_STORE/runs/rloo_muson_ego_cap1024_video400/predictions.jsonl")
test "$N" -eq 400
