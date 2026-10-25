#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --job-name=social-9b-smoke
#SBATCH --output=logs/smoke-%j.log
#SBATCH --error=logs/smoke-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
nvidia-smi
python scripts/smoke.py
