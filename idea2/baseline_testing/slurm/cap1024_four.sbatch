#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --array=0-3%1
#SBATCH --job-name=social-cap1024
#SBATCH --output=logs/cap1024-%A_%a.log
#SBATCH --error=logs/cap1024-%A_%a.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
case "$SLURM_ARRAY_TASK_ID" in
  0) KIND=base; ADAPTER_ARGS=() ;;
  1) KIND=rloo; ADAPTER_ARGS=(--adapter "$SOCIAL_RL_STORE/runs/rloo/checkpoint-00100") ;;
  2) KIND=grpo; ADAPTER_ARGS=(--adapter "$SOCIAL_RL_STORE/runs/grpo/checkpoint-00100") ;;
  3) KIND=sft; ADAPTER_ARGS=(--adapter "$SOCIAL_RL_STORE/runs/sft/checkpoint-00100") ;;
  *) echo "Unknown array task $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac
OUTPUT="$SOCIAL_RL_STORE/runs/${KIND}_ego_cap1024_video400"
python scripts/evaluate.py \
  --manifest "$SOCIAL_RL_STORE/data/manifests/ego_midterm_panel_accessible.jsonl" \
  --output "$OUTPUT" "${ADAPTER_ARGS[@]}" \
  --conditions video --max-new-tokens 1024 --max-hours 5.5
N=$(wc -l < "$OUTPUT/predictions.jsonl")
if [[ "$N" -ne 400 ]]; then
  echo "Incomplete $KIND cap1024 run: $N/400; resubmit this array index to resume" >&2
  exit 1
fi
