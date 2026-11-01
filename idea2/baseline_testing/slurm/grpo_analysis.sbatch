#!/bin/bash
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=2
#SBATCH --mem=3800M
#SBATCH --time=00:30:00
#SBATCH --job-name=social-grpo-analysis
#SBATCH --output=logs/grpo-analysis-%j.log
#SBATCH --error=logs/grpo-analysis-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
python scripts/summarize.py \
  --manifest "$SOCIAL_RL_STORE/data/manifests/r3_test_accessible.jsonl" \
  --runs "$SOCIAL_RL_STORE/runs/base_r3_test" \
         "$SOCIAL_RL_STORE/runs/rloo_r3_test" \
         "$SOCIAL_RL_STORE/runs/sft_r3_test" \
         "$SOCIAL_RL_STORE/runs/grpo_r3_test" \
  --output reports/generated/grpo_r3_test
python scripts/summarize.py \
  --manifest "$SOCIAL_RL_STORE/data/manifests/ego_midterm_panel_accessible.jsonl" \
  --runs "$SOCIAL_RL_STORE/runs/base_ego_video400" \
         "$SOCIAL_RL_STORE/runs/rloo_ego_video400" \
         "$SOCIAL_RL_STORE/runs/sft_ego_video400" \
         "$SOCIAL_RL_STORE/runs/grpo_ego_video400" \
  --output reports/generated/grpo_ego_video400
python scripts/failure_counts.py \
  --manifest "$SOCIAL_RL_STORE/data/manifests/ego_midterm_panel_accessible.jsonl" \
  --run "$SOCIAL_RL_STORE/runs/grpo_ego_video400" \
  --output reports/generated/grpo_ego_failures
