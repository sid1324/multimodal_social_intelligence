#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --job-name=extra-r3-visual
#SBATCH --output=logs/extra-r3-visual-%j.log
#SBATCH --error=logs/extra-r3-visual-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
python scripts/evaluate.py --manifest "$SOCIAL_RL_STORE/data/manifests/ego_midterm_panel_accessible.jsonl" --adapter "$SOCIAL_RL_STORE/runs/r3_sft_extra3032/checkpoint-03032" --output "$SOCIAL_RL_STORE/runs/extra_r3_visual_controls100" --conditions blind mismatch --blind-same-prompt --limit 100 --max-new-tokens 1024 --max-hours 3.5
