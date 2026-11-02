#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --job-name=psy-images-smoke
#SBATCH --output=logs/psy-images-smoke-%j.log
#SBATCH --error=logs/psy-images-smoke-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
PYTHON=python
if [ "$1" = 27b ]; then PYTHON="$SOCIAL_RL_STORE/envs/qwen38/bin/python"; fi
"$PYTHON" scripts/smoke_frame_images.py --family "$1"
