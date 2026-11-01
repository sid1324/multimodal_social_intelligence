#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --job-name=psy-sft
#SBATCH --output=logs/psy-sft-%j.log
#SBATCH --error=logs/psy-sft-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
python scripts/train_psychemme_sft.py "$@"
