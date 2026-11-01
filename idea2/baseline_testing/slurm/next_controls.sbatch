#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=04:30:00
#SBATCH --job-name=next-controls
#SBATCH --output=logs/next-controls-%j.log
#SBATCH --error=logs/next-controls-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
if [ "$1" = r3 ]; then
 ADAPTER="$SOCIAL_RL_STORE/runs/r3_sft_extra3032/checkpoint-03032"
 test -f "$ADAPTER/adapter_config.json"
 python scripts/evaluate.py --manifest "$SOCIAL_RL_STORE/data/manifests/ego_midterm_panel_accessible.jsonl" --adapter "$ADAPTER" --output "$SOCIAL_RL_STORE/runs/r3_sft_extra3032/ego1024_video400" --conditions video --max-new-tokens 1024 --max-hours 3.5
else
 python scripts/evaluate.py --manifest "$SOCIAL_RL_STORE/data/manifests/ego_midterm_panel_accessible.jsonl" --output "$SOCIAL_RL_STORE/runs/base_visual_controls100" --conditions blind mismatch --blind-same-prompt --limit 100 --max-new-tokens 1024 --max-hours 3.5
fi
