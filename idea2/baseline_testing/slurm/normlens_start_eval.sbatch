#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --job-name=social-normlens-start-eval
#SBATCH --output=logs/normlens-start-eval-%j.log
#SBATCH --error=logs/normlens-start-eval-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
python scripts/evaluate_normlens.py \
  --manifest "$SOCIAL_RL_STORE/data/normlens/val.jsonl" \
  --output "$SOCIAL_RL_STORE/runs/rloo_normlens_start_val" \
  --adapter "$SOCIAL_RL_STORE/runs/rloo/checkpoint-00100" --max-hours 0.8
N=$(wc -l < "$SOCIAL_RL_STORE/runs/rloo_normlens_start_val/predictions.jsonl")
test "$N" -eq 123
