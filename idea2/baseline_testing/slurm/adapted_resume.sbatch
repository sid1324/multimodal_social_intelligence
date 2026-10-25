#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --job-name=social-rl-eval-resume
#SBATCH --output=logs/adapted-resume-%j.log
#SBATCH --error=logs/adapted-resume-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
ADAPTER_PATH=$(python -c 'import json; from social_rl.common import STORE; print(json.load(open(STORE/"runs/rloo/latest.json"))["checkpoint"])')
python scripts/evaluate.py --manifest "$SOCIAL_RL_STORE/data/manifests/ego_midterm_matched.jsonl" \
  --output "$SOCIAL_RL_STORE/runs/rloo_ego" --adapter "$ADAPTER_PATH" \
  --conditions video blind mismatch --max-hours 2.2
python scripts/summarize.py --manifest "$SOCIAL_RL_STORE/data/manifests/ego_midterm_panel_accessible.jsonl" \
  --runs "$SOCIAL_RL_STORE/runs/base_ego" "$SOCIAL_RL_STORE/runs/rloo_ego" \
  --output reports/generated/comparison
