#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --job-name=psy-src-eval
#SBATCH --output=logs/psy-src-%j.log
#SBATCH --error=logs/psy-src-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
START="$1"
ADAPTER="$SOCIAL_RL_STORE/runs/psychemme_sft_${START}/checkpoint-00300"
test -f "$ADAPTER/adapter_config.json"
python scripts/evaluate_psychemme_sft.py --adapter "$ADAPTER" \
  --output "$SOCIAL_RL_STORE/runs/psychemme_sft_${START}_source_val120" --limit 120
