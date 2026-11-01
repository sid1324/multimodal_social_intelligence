#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=04:30:00
#SBATCH --job-name=psy-eval
#SBATCH --output=logs/psy-eval-%j.log
#SBATCH --error=logs/psy-eval-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
START="$1"
ADAPTER="$SOCIAL_RL_STORE/runs/psychemme_sft_${START}/checkpoint-00300"
test -f "$ADAPTER/adapter_config.json"
python scripts/evaluate.py \
  --manifest "$SOCIAL_RL_STORE/data/manifests/ego_midterm_panel_accessible.jsonl" \
  --output "$SOCIAL_RL_STORE/runs/psychemme_sft_${START}_ego1024_video400" \
  --adapter "$ADAPTER" --conditions video --max-new-tokens 1024 --max-hours 4.0
test "$(wc -l < "$SOCIAL_RL_STORE/runs/psychemme_sft_${START}_ego1024_video400/predictions.jsonl")" -eq 400
