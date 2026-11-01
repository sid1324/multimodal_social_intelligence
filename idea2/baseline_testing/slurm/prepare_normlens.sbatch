#!/bin/bash
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --account=cis260338p
#SBATCH --cpus-per-task=2
#SBATCH --mem=3G
#SBATCH --time=00:30:00
#SBATCH --job-name=social-normlens-prepare
#SBATCH --output=logs/normlens-prepare-%j.log
#SBATCH --error=logs/normlens-prepare-%j.log
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
"$SOCIAL_RL_STORE/envs/data/bin/python" scripts/prepare_normlens.py
