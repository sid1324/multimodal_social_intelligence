#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --job-name=social-normlens-smoke
#SBATCH --output=logs/normlens-smoke-%j.log
#SBATCH --error=logs/normlens-smoke-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
python scripts/evaluate_normlens.py \
  --manifest "$SOCIAL_RL_STORE/data/normlens/val.jsonl" \
  --output "$SOCIAL_RL_STORE/runs/rloo_normlens_smoke" \
  --adapter "$SOCIAL_RL_STORE/runs/rloo/checkpoint-00100" --limit 2 --max-hours 0.2
N=$(wc -l < "$SOCIAL_RL_STORE/runs/rloo_normlens_smoke/predictions.jsonl")
test "$N" -eq 2
