#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=01:30:00
#SBATCH --job-name=q38-smoke
#SBATCH --output=logs/q38-smoke-%j.log
#SBATCH --error=logs/q38-smoke-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
/ocean/projects/cis260338p/apatawar/social_norm_rl/envs/qwen38/bin/python scripts/qwen38_smoke.py
