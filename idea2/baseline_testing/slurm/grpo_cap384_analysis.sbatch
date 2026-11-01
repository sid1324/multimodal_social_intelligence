#!/bin/bash
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=2
#SBATCH --mem=3800M
#SBATCH --time=00:30:00
#SBATCH --job-name=social-grpo-cap384-analysis
#SBATCH --output=logs/grpo-cap384-analysis-%j.log
#SBATCH --error=logs/grpo-cap384-analysis-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
python scripts/summarize.py \
  --manifest "$SOCIAL_RL_STORE/data/manifests/ego_midterm_panel_accessible.jsonl" \
  --runs "$SOCIAL_RL_STORE/runs/base_ego_cap384_video400" \
         "$SOCIAL_RL_STORE/runs/rloo_ego_cap384_video400" \
         "$SOCIAL_RL_STORE/runs/sft_ego_cap384_video400" \
         "$SOCIAL_RL_STORE/runs/grpo_ego_cap384_video400" \
  --output reports/generated/grpo_ego_cap384_video400
python scripts/summarize.py \
  --manifest "$SOCIAL_RL_STORE/data/manifests/ego_midterm_panel_accessible.jsonl" \
  --runs "$SOCIAL_RL_STORE/runs/grpo_ego_video400" \
         "$SOCIAL_RL_STORE/runs/grpo_ego_cap384_video400" \
  --output reports/generated/grpo_cap_effect
python scripts/failure_counts.py \
  --manifest "$SOCIAL_RL_STORE/data/manifests/ego_midterm_panel_accessible.jsonl" \
  --run "$SOCIAL_RL_STORE/runs/grpo_ego_cap384_video400" \
  --output reports/generated/grpo_cap384_failures
