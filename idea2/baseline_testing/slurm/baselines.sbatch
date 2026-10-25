#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --job-name=social-midterm-base
#SBATCH --output=logs/baselines-%j.log
#SBATCH --error=logs/baselines-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
python scripts/evaluate.py --manifest "$SOCIAL_RL_STORE/data/manifests/ego_midterm_panel_accessible.jsonl" \
  --output "$SOCIAL_RL_STORE/runs/base_ego" --conditions video blind mismatch --max-hours 2.2
python scripts/evaluate.py --manifest "$SOCIAL_RL_STORE/data/manifests/r3_val_accessible.jsonl" \
  --output "$SOCIAL_RL_STORE/runs/base_r3_val" --conditions video --limit 40 --max-hours 0.4
python scripts/evaluate.py --manifest "$SOCIAL_RL_STORE/data/manifests/r3_test_accessible.jsonl" \
  --output "$SOCIAL_RL_STORE/runs/base_r3_test" --conditions video blind --limit 60 --max-hours 0.6
python scripts/summarize.py --manifest "$SOCIAL_RL_STORE/data/manifests/ego_midterm_panel_accessible.jsonl" \
  --runs "$SOCIAL_RL_STORE/runs/base_ego" --output reports/generated/baseline
python scripts/error_review.py --manifest "$SOCIAL_RL_STORE/data/manifests/ego_midterm_panel_accessible.jsonl" \
  --run "$SOCIAL_RL_STORE/runs/base_ego" --output reports/generated/error_review
