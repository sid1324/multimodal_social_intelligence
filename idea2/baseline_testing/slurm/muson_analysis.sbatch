#!/bin/bash
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=2
#SBATCH --mem=3800M
#SBATCH --time=00:30:00
#SBATCH --job-name=social-muson-analysis
#SBATCH --output=logs/muson-analysis-%j.log
#SBATCH --error=logs/muson-analysis-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
python scripts/summarize.py \
  --manifest "$SOCIAL_RL_STORE/data/manifests/ego_midterm_panel_accessible.jsonl" \
  --runs "$SOCIAL_RL_STORE/runs/base_ego_cap1024_video400" \
         "$SOCIAL_RL_STORE/runs/rloo_ego_cap1024_video400" \
         "$SOCIAL_RL_STORE/runs/sft_ego_cap1024_video400" \
         "$SOCIAL_RL_STORE/runs/rloo_muson_ego_cap1024_video400" \
  --output reports/generated/muson_transfer_ego1024
python scripts/summarize.py \
  --manifest "$SOCIAL_RL_STORE/data/manifests/ego_midterm_panel_accessible.jsonl" \
  --runs "$SOCIAL_RL_STORE/runs/rloo_ego_cap1024_video400" \
         "$SOCIAL_RL_STORE/runs/rloo_muson_ego_cap1024_video400" \
  --output reports/generated/muson_transfer_paired_rloo
python scripts/failure_counts.py \
  --manifest "$SOCIAL_RL_STORE/data/manifests/ego_midterm_panel_accessible.jsonl" \
  --run "$SOCIAL_RL_STORE/runs/rloo_muson_ego_cap1024_video400" \
  --output reports/generated/muson_transfer_failures
