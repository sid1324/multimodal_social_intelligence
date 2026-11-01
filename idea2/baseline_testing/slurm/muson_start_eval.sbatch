#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --job-name=social-muson-start-eval
#SBATCH --output=logs/muson-start-eval-%j.log
#SBATCH --error=logs/muson-start-eval-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
python scripts/evaluate_muson.py \
  --manifest "$SOCIAL_RL_STORE/data/muson/val.jsonl" \
  --output "$SOCIAL_RL_STORE/runs/rloo_muson_start_val300" \
  --adapter "$SOCIAL_RL_STORE/runs/rloo/checkpoint-00100" --limit 300 --max-hours 1.7
N=$(wc -l < "$SOCIAL_RL_STORE/runs/rloo_muson_start_val300/predictions.jsonl")
test "$N" -eq 300
