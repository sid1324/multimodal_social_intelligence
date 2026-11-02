#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --job-name=visual-resume
#SBATCH --output=logs/visual-resume-%j.log
#SBATCH --error=logs/visual-resume-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
RUN="$1"
ADAPTER=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["checkpoint"])' "$RUN/selected.json")
python scripts/evaluate.py --manifest "$SOCIAL_RL_STORE/data/manifests/ego_midterm_panel_accessible.jsonl" --adapter "$ADAPTER" --output "$RUN/visual_controls100" --conditions blind mismatch --blind-same-prompt --limit 100 --max-new-tokens 1024 --max-hours 3.5
