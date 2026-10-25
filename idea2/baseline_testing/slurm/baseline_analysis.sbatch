#!/bin/bash
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=2
#SBATCH --mem=3800M
#SBATCH --time=00:30:00
#SBATCH --job-name=social-baseline-analysis
#SBATCH --output=logs/baseline-analysis-%j.log
#SBATCH --error=logs/baseline-analysis-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
python scripts/failure_counts.py \
  --manifest "$SOCIAL_RL_STORE/data/manifests/ego_midterm_panel_accessible.jsonl" \
  --run "$SOCIAL_RL_STORE/runs/base_ego" --output reports/generated/baseline
python scripts/summarize.py \
  --manifest "$SOCIAL_RL_STORE/data/manifests/r3_val_accessible.jsonl" \
  --runs "$SOCIAL_RL_STORE/runs/base_r3_val" --output reports/generated/r3_base_val
python scripts/summarize.py \
  --manifest "$SOCIAL_RL_STORE/data/manifests/r3_test_accessible.jsonl" \
  --runs "$SOCIAL_RL_STORE/runs/base_r3_test" --output reports/generated/r3_base_test
