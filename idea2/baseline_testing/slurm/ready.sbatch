#!/bin/bash
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=08:00:00
#SBATCH --job-name=social-model-ready
#SBATCH --output=logs/ready-%j.log
#SBATCH --error=logs/ready-%j.log
set -euo pipefail
STORE=${SOCIAL_RL_STORE:-/ocean/projects/cis260338p/apatawar/social_norm_rl}
for ((attempt=0; attempt<920; attempt++)); do
  if [[ -f "$STORE/envs/qwen35/READY" && -f "$STORE/models/Qwen3.5-9B/READY" &&
        -s "$STORE/data/manifests/r3_train_accessible.jsonl" ]]; then
    echo "Verified model, environment, and R3 smoke input ready at $(date)"
    exit 0
  fi
  sleep 30
done
echo "Timed out waiting for verified model and environment" >&2
exit 1
