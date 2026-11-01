#!/bin/bash
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=2
#SBATCH --mem=3800M
#SBATCH --time=00:30:00
#SBATCH --job-name=psy-epoch-analysis
#SBATCH --output=logs/psy-epoch-analysis-%j.log
#SBATCH --error=logs/psy-epoch-analysis-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
python scripts/summarize.py --manifest "$SOCIAL_RL_STORE/data/manifests/ego_midterm_panel_accessible.jsonl" --runs \
 "$SOCIAL_RL_STORE/runs/sft_ego_cap1024_video400" \
 "$SOCIAL_RL_STORE/runs/psychemme_sft_r3sft_ego1024_video400" \
 "$SOCIAL_RL_STORE/runs/psychemme_epoch_base_s777/ego1024_video400" \
 "$SOCIAL_RL_STORE/runs/psychemme_epoch_base_s1777/ego1024_video400" \
 "$SOCIAL_RL_STORE/runs/psychemme_epoch_r3sft_s777/ego1024_video400" \
 "$SOCIAL_RL_STORE/runs/psychemme_epoch_r3sft_s1777/ego1024_video400" \
 "$SOCIAL_RL_STORE/runs/r3_sft_extra3032/ego1024_video400" \
 --output reports/generated/psychemme_epoch_results
