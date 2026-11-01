#!/bin/bash
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=2
#SBATCH --mem=3800M
#SBATCH --time=00:30:00
#SBATCH --job-name=social-cap1024-analysis
#SBATCH --output=logs/cap1024-analysis-%j.log
#SBATCH --error=logs/cap1024-analysis-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
MANIFEST="$SOCIAL_RL_STORE/data/manifests/ego_midterm_panel_accessible.jsonl"
RUNS=()
for KIND in base rloo grpo sft; do
  RUN="$SOCIAL_RL_STORE/runs/${KIND}_ego_cap1024_video400"
  N=$(wc -l < "$RUN/predictions.jsonl")
  if [[ "$N" -ne 400 ]]; then
    echo "Incomplete $KIND run: $N/400" >&2
    exit 1
  fi
  RUNS+=("$RUN")
  python scripts/failure_counts.py --manifest "$MANIFEST" --run "$RUN" \
    --output "reports/generated/cap1024_${KIND}_failures"
  python scripts/summarize.py --manifest "$MANIFEST" \
    --runs "$SOCIAL_RL_STORE/runs/${KIND}_ego_cap384_video400" "$RUN" \
    --output "reports/generated/cap1024_${KIND}_cap_effect"
done
python scripts/summarize.py --manifest "$MANIFEST" --runs "${RUNS[@]}" \
  --output reports/generated/cap1024_video400
