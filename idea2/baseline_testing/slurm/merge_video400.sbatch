#!/bin/bash
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=2
#SBATCH --mem=3800M
#SBATCH --time=00:30:00
#SBATCH --job-name=social-video400-merge
#SBATCH --output=logs/video400-merge-%j.log
#SBATCH --error=logs/video400-merge-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
python scripts/merge_video400.py
python scripts/summarize.py --manifest "$SOCIAL_RL_STORE/data/manifests/ego_midterm_panel_accessible.jsonl" \
  --runs "$SOCIAL_RL_STORE/runs/base_ego_video400" "$SOCIAL_RL_STORE/runs/rloo_ego_video400" \
  "$SOCIAL_RL_STORE/runs/sft_ego_video400" --output reports/generated/video400
for KIND in base rloo sft; do
  python scripts/failure_counts.py --manifest "$SOCIAL_RL_STORE/data/manifests/ego_midterm_panel_accessible.jsonl" \
    --run "$SOCIAL_RL_STORE/runs/${KIND}_ego_video400" --output "reports/generated/video400_${KIND}_failures"
done
