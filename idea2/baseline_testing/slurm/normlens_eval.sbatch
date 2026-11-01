#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=04:30:00
#SBATCH --job-name=social-normlens-eval
#SBATCH --output=logs/normlens-eval-%j.log
#SBATCH --error=logs/normlens-eval-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
ADAPTER="$SOCIAL_RL_STORE/runs/rloo_normlens/checkpoint-00025"
python scripts/evaluate_normlens.py \
  --manifest "$SOCIAL_RL_STORE/data/normlens/val.jsonl" \
  --output "$SOCIAL_RL_STORE/runs/rloo_normlens_val" \
  --adapter "$ADAPTER" --max-hours 0.8
python scripts/evaluate.py \
  --manifest "$SOCIAL_RL_STORE/data/manifests/ego_midterm_panel_accessible.jsonl" \
  --output "$SOCIAL_RL_STORE/runs/rloo_normlens_ego_cap1024_video400" \
  --adapter "$ADAPTER" --conditions video --max-new-tokens 1024 --max-hours 3.2
N=$(wc -l < "$SOCIAL_RL_STORE/runs/rloo_normlens_ego_cap1024_video400/predictions.jsonl")
test "$N" -eq 400
