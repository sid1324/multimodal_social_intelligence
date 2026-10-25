#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --job-name=social-video-extension
#SBATCH --output=logs/extend-video-%j.log
#SBATCH --error=logs/extend-video-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
KIND=${1:?Specify base, rloo, or sft}
case "$KIND" in
  base) ADAPTER_ARGS=() ;;
  rloo|sft)
    ADAPTER_PATH=$(python -c 'import json,sys; from social_rl.common import STORE; print(json.load(open(STORE/"runs"/sys.argv[1]/"latest.json"))["checkpoint"])' "$KIND")
    ADAPTER_ARGS=(--adapter "$ADAPTER_PATH") ;;
  *) echo "Unknown model kind: $KIND" >&2; exit 2 ;;
esac
python scripts/evaluate.py --manifest "$SOCIAL_RL_STORE/data/manifests/ego_midterm_remaining260.jsonl" \
  --output "$SOCIAL_RL_STORE/runs/${KIND}_ego_video_remaining" \
  "${ADAPTER_ARGS[@]}" --conditions video --max-hours 3.5
