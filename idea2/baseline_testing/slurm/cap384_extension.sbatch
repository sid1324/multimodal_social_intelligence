#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --job-name=social-cap384-extension
#SBATCH --output=logs/cap384-extension-%j.log
#SBATCH --error=logs/cap384-extension-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
KIND=${1:?Specify base, rloo, or sft}
case "$KIND" in
  base)
    ADAPTER_ARGS=()
    MANIFEST="$SOCIAL_RL_STORE/data/manifests/ego_midterm_remaining360.jsonl"
    EXPECTED=360 ;;
  rloo|sft)
    ADAPTER_PATH=$(python -c 'import json,sys; from social_rl.common import STORE; print(json.load(open(STORE/"runs"/sys.argv[1]/"latest.json"))["checkpoint"])' "$KIND")
    ADAPTER_ARGS=(--adapter "$ADAPTER_PATH")
    if [[ "$KIND" == rloo ]]; then
      MANIFEST="$SOCIAL_RL_STORE/data/manifests/ego_midterm_remaining360.jsonl"
      EXPECTED=360
    else
      MANIFEST="$SOCIAL_RL_STORE/data/manifests/ego_midterm_panel_accessible.jsonl"
      EXPECTED=400
    fi ;;
  *) echo "Unknown model kind: $KIND" >&2; exit 2 ;;
esac
OUTPUT="$SOCIAL_RL_STORE/runs/${KIND}_ego_cap384_extension"
python scripts/evaluate.py --manifest "$MANIFEST" --output "$OUTPUT" \
  "${ADAPTER_ARGS[@]}" --conditions video --max-new-tokens 384 --max-hours 3.5
N=$(wc -l < "$OUTPUT/predictions.jsonl")
if [[ "$N" -ne "$EXPECTED" ]]; then
  echo "Incomplete $KIND cap384 run: $N/$EXPECTED" >&2
  exit 1
fi
