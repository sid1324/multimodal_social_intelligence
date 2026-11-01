#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --job-name=social-normlens-r3-eval
#SBATCH --output=logs/normlens-r3-eval-%j.log
#SBATCH --error=logs/normlens-r3-eval-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
python scripts/evaluate.py \
  --manifest "$SOCIAL_RL_STORE/data/manifests/r3_test_accessible.jsonl" \
  --output "$SOCIAL_RL_STORE/runs/rloo_normlens_r3_test" \
  --adapter "$SOCIAL_RL_STORE/runs/rloo_normlens/checkpoint-00025" \
  --conditions video blind --max-hours 1.1
N=$(wc -l < "$SOCIAL_RL_STORE/runs/rloo_normlens_r3_test/predictions.jsonl")
EXPECTED=$((2 * $(wc -l < "$SOCIAL_RL_STORE/data/manifests/r3_test_accessible.jsonl")))
test "$N" -eq "$EXPECTED"
